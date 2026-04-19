# Pruebas de construcción del grafo — estado actual

## Qué hemos construido


El sistema se divide en dos grandes bloques: el pipeline de ingesta, que construye el grafo, y la capa de recuperación, que lo consulta.

### Pipeline de ingesta

El pipeline parte de los textos en bruto de Project Gutenberg y termina con un grafo Neo4j poblado. Tiene cuatro etapas.

**Descarga y preparación del texto.** El TextProcessor descarga los ficheros de texto de Project Gutenberg (cinco colecciones: Adventures, Memoirs, Return, His Last Bow y Casebook), elimina las cabeceras y pies de página estándar de Gutenberg y separa cada colección en relatos individuales mediante expresiones regulares que detectan los encabezados de capítulo. Actualmente trabajamos con 9 relatos de la Fase 1.

**Chunking con overlap.** Cada relato se divide en fragmentos de aproximadamente 1500 caracteres respetando los límites de párrafo, es decir, nunca se corta un párrafo por la mitad. Entre fragmentos consecutivos se aplica un overlap de 200 caracteres tomando párrafos completos del final del chunk anterior, para que el contexto no se pierda en los bordes. A cada chunk se le asigna su posición narrativa dentro del relato (beginning, middle, end). El resultado son 321 chunks para los 9 relatos. Cada chunk se vectoriza con el modelo de embeddings de Vertex AI y se almacena en Neo4j como nodo Chunk con su embedding, lo que permite búsqueda vectorial por similitud semántica.

**Extracción multipaso de entidades y relaciones.** Esta es la parte central del pipeline. Para cada chunk, el EntityExtractor lanza dos llamadas a Gemini Flash.

En el primer paso extrae entidades tipadas: personajes (Character), lugares (Location), crímenes (Crime), objetos relevantes para la trama (Object), deducciones de Holmes (Deduction), escenas narrativas (Scene) y eventos clave (Event). Para ayudar a resolver referencias ambiguas como pronombres o alias, se mantiene un contexto narrativo deslizante (NarrativeContext) que acumula los personajes presentes, la ubicación actual y los eventos recientes a lo largo del relato.

En el segundo paso, con la lista de entidades ya identificadas, extrae las relaciones entre ellas. Los tipos de relación que el sistema modela son: APPEARS_IN (personaje aparece en relato), KNOWS (relación entre personajes), INVESTIGATES (personaje investiga un crimen), USES (personaje usa un objeto), FOUND_AT (objeto encontrado en lugar), LIVES_AT (personaje reside en lugar), PRESENT_IN (personaje presente en escena), TAKES_PLACE_IN (escena ocurre en lugar), FOLLOWS (una escena sigue a otra), BASED_ON (deducción basada en objeto o evento), LEADS_TO (una deducción conduce a otra), PARTICIPATES_IN (personaje participa en evento) y OCCURS_IN (crimen ocurre en lugar).

Tras procesar todos los chunks de un relato, se realiza una entity resolution que agrupa variantes del mismo nombre en un único nodo canónico. Para personajes se usan reglas lingüísticas (strip de títulos honoríficos como Mr., Dr., Colonel, y comparación de sufijos y prefijos de palabras). Para el resto de tipos se usan embeddings con un umbral de similitud coseno de 0.92, con una zona gris entre 0.80 y 0.92 donde se consulta al LLM para confirmar si son la misma entidad.

Finalmente hay una fase de normalización cross-story que unifica los nombres canónicos de personajes entre relatos distintos, para que Holmes tenga el mismo nombre en todos ellos y Neo4j no cree nodos duplicados.

**Carga en Neo4j.** El Neo4jManager toma los resultados de extracción y los escribe en la base de datos con operaciones MERGE, que son idempotentes: si el nodo ya existe lo actualiza, si no lo crea. Las relaciones se crean con MERGE también, y en los casos donde el LLM puede haber parafraseado un nombre (por ejemplo el título de una escena), se aplica fuzzy matching por similitud Jaccard antes de buscar el nodo destino.

### Estado actual del grafo

El grafo contiene 9 relatos, 321 chunks, 165 personajes, 218 ubicaciones, 223 deducciones, 543 eventos, 521 objetos, 254 escenas y 79 crímenes. El total de relaciones supera las 4.700 instancias distribuidas en 16 tipos. Sherlock Holmes es el nodo más conectado con 711 relaciones.

---

## Lo que ha funcionado

El esquema de nodos y relaciones es semánticamente correcto y permite consultas estructuradas que serían imposibles con recuperación vectorial pura, como trazar cadenas de deducción de Holmes o encontrar todos los personajes que comparten escena con el antagonista. La separación entre paso de entidades y paso de relaciones ha demostrado ser importante: cuando se intentaba extraer todo a la vez el LLM producía más errores y entidades inventadas.

La entity resolution de personajes funciona bien para los casos frecuentes. El filtro de entidades genéricas elimina pronombres y roles sueltos que el LLM extrae por error. La normalización cross-story garantiza que Holmes y Watson sean el mismo nodo en todos los relatos.

El pipeline completo corre de forma estable y es reproducible: el checkpoint de extracción se guarda en JSON para no depender de que la API esté disponible en cada ejecución.

---

## Lo que no ha funcionado

**Seis tipos de relación con cero instancias en la primera versión.** PRESENT_IN, TAKES_PLACE_IN, FOLLOWS, PARTICIPATES_IN, INVESTIGATES y OCCURS_IN no generaban ninguna relación en Neo4j. El problema era que el código buscaba los nodos por ID interno generado (por ejemplo silver_blaze_scene_0) mientras que el LLM usaba el título descriptivo de la escena. Al unificar las claves de búsqueda para usar siempre el nombre o título del nodo, todos los tipos empezaron a poblarse.

**El LLM inventaba nombres de escenas y eventos en las relaciones.** Aunque la extracción de entidades producía títulos correctos, al extraer relaciones el modelo los parafraseaba. Añadir una instrucción explícita en el prompt para que solo usara nombres del listado de entidades ya extraídas redujo el problema, aunque no lo eliminó completamente. El fuzzy matching aplicado en la carga absorbe las variaciones menores que siguen ocurriendo.

**BASED_ON solo almacenaba relaciones hacia Object.** El prompt indicaba que BASED_ON podía apuntar a Object o a Event, pero el código solo buscaba nodos Object. Las relaciones cuyo destino era un Event, Character, Crime o Location se perdían en silencio. Se corrigió con una query multi-etiqueta que prueba todos los tipos posibles.

**Fallo silencioso de API durante la extracción de Blue Carbuncle.** La cuota de Gemini se agotó durante el procesamiento del séptimo relato. El extractor capturaba la excepción y continuaba devolviendo resultados vacíos sin ningún aviso, así que el relato quedó con una sola deducción, dos ubicaciones y personajes de otros relatos alucinados por el LLM. Se añadió detección de chunks vacíos con warnings, pero el relato tampoco mejoró en la segunda ejecución completa. Decidimos eliminarlo del conjunto de desarrollo.

**Fragmentación de Holmes y Watson en las primeras versiones.** Holmes aparecía como tres nodos distintos (Holmes, Mr. Sherlock Holmes, Mister Sherlock Holmes) y Watson como dos. Se resolvió ajustando las reglas de entity resolution para normalizar títulos honoríficos antes de comparar nombres.

---

## Problemas abiertos y puntos donde pedimos feedback


**30% de deducciones sin conectar en Neo4j.** Las relaciones BASED_ON que apuntan a entidades que el LLM referencia pero no extrae como nodos se pierden inevitablemente porque no hay nodo destino. Podríamos restringir el prompt para que BASED_ON solo apunte a Object, eliminando el ruido pero también información válida, o podríamos añadir un paso de extracción adicional para recuperar esas entidades faltantes. No sabemos qué es más pragmático de cara a la fase de retrieval.

**Alucinaciones de personajes en algunos relatos.** En ciertos chunks el LLM incluye personajes del canon de Holmes que no aparecen en el texto de ese relato concreto. El filtro de ruido actual elimina pronombres y roles genéricos pero no detecta nombres propios de otros relatos. No sabemos si este ruido afectará significativamente a la calidad del retrieval o si es un problema que se puede ignorar en esta fase.

**Volumen de Objects y Events.** Con 521 objetos y 543 eventos para 9 relatos, sospechamos que la entity resolution no está fusionando bien entidades no-personaje con nombres distintos que refieren a lo mismo. No hemos priorizado esto porque los problemas estructurales de relaciones eran más urgentes. No sabemos si este volumen es un problema real para el retrieval o si es aceptable.
