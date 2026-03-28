# StrategyEngine AI - Blueprint de Despliegue Enterprise

## Objetivo

Este documento describe el patrón de despliegue recomendado para vender StrategyEngine AI a empresas de forma profesional.

Responde a:

- cómo debe ejecutarse el producto dentro de la infraestructura del cliente
- qué debe desplegarse primero en pilotos
- qué debe endurecerse para producción
- cómo explicar la arquitectura durante ventas o validación técnica

## Posicionamiento Comercial

StrategyEngine AI debe venderse como:

**un appliance enterprise self-hosted para flujos de datos a modelo**

El cliente:

- lo despliega en su entorno
- configura API keys y backends de ejecución desde la UI
- mantiene datasets, runs, logs y artefactos bajo su control

## Niveles de Entrega Recomendados

### Nivel 1 - Piloto

Recomendado para:
- pequeñas y medianas empresas
- primer proof of value
- ciclos de venta cortos

Topología:
- un contenedor principal
- volúmenes persistentes montados
- worker local como subprocess
- sandbox local o gateway remoto

Objetivo operativo:
- el despliegue más simple posible que ya se perciba como producto real

### Nivel 2 - Producción Self-Hosted

Recomendado para:
- entornos con revisión de seguridad
- varios usuarios
- adopción a medio y largo plazo

Topología:
- servicio dedicado de UI
- servicio dedicado de worker
- stores persistentes de metadatos y artefactos
- sandbox gateway aislado
- logs y monitorización centralizados

Objetivo operativo:
- fiabilidad de producción, auditabilidad y preparación para revisión enterprise

### Nivel 3 - Despliegue Privado Gestionado

Recomendado para:
- clientes que no quieren operar la plataforma

Topología:
- misma arquitectura que el self-hosted
- operada en un tenant o cuenta cloud dedicada

Objetivo operativo:
- reducir carga operativa sin sacrificar aislamiento

## Componentes Mínimos Desplegables

### 1. UI / Plano de Control

Base actual:
- [app.py](C:/Users/santi/Projects/Hackathon_Gemini_Agents/app.py)

Responsabilidades:
- aceptar inputs
- renderizar informes
- configurar proveedores y modelos
- mostrar historial y progreso de runs

### 2. Worker / Orquestador

Base actual:
- [src/utils/background_worker.py](C:/Users/santi/Projects/Hackathon_Gemini_Agents/src/utils/background_worker.py)
- [src/graph/graph.py](C:/Users/santi/Projects/Hackathon_Gemini_Agents/src/graph/graph.py)

Responsabilidades:
- ejecutar runs fuera de la sesión del navegador
- coordinar el pipeline multiagente
- persistir estado, logs y outputs finales

### 3. Backend de Ejecución en Sandbox

Responsabilidades:
- aislar el código generado
- aplicar límites de tiempo y recursos
- devolver ficheros, logs y estado de salida

Modelo recomendado de cara al cliente:
- un contrato estándar de sandbox gateway

Referencia:
- [SANDBOX_GATEWAY.md](C:/Users/santi/Projects/Hackathon_Gemini_Agents/SANDBOX_GATEWAY.md)

### 4. Almacenamiento Persistente

Usado para:
- runs
- artefactos
- PDFs
- informes intermedios
- reproducibilidad

### 5. Configuración y Secretos

Usado para:
- API keys de LLM
- credenciales del sandbox
- routing de modelos
- configuración del backend de ejecución

Dirección ya alineada en el producto:
- configuración desde la UI
- sin edición manual de archivos internos por parte del cliente

## Patrones de Despliegue Recomendados

## A. Piloto con Docker Compose

Úsalo cuando:
- el cliente quiere algo rápido de levantar
- la revisión de seguridad es ligera
- el equipo es pequeño

Forma recomendada:

```text
docker-compose
  - strategyengine-ui-worker
  - volumen ./runs
  - volumen ./data
  - volumen ./artifacts
```

Notas:
- válido para piloto
- no es la forma final para despliegues enterprise grandes
- el worker puede seguir siendo subprocess en esta etapa

## B. Producción en la Infraestructura del Cliente

Úsalo cuando:
- el cliente tiene IT/SecOps implicado
- hay varios usuarios
- la gobernanza de datos importa

Forma recomendada:

```text
Load Balancer
  -> servicio UI / Control Plane
  -> servicio Worker
  -> almacenamiento de artefactos
  -> base de datos de metadatos
  -> almacén de secretos
  -> sandbox gateway
```

Beneficios:
- escalado más limpio
- mejor observabilidad
- mejor separación de responsabilidades
- revisión de seguridad más sencilla

## Recomendación de Límite de Seguridad

La recomendación más importante es:

**ejecutar el sandbox dentro del perímetro de infraestructura del cliente**

Eso significa:
- ejecución local para casos de baja sensibilidad
- o gateway remoto de sandbox alojado por el cliente

Es la respuesta correcta cuando un cliente pregunta:
- dónde corre el código
- dónde viven los datos
- quién paga y controla el compute

## Respuesta de Referencia para Clientes

Si el cliente pregunta "¿cómo se integraría esto con nuestra infraestructura?", la respuesta profesional es:

1. StrategyEngine AI se despliega dentro de vuestro entorno como aplicación web más worker.
2. Vuestro equipo configura API keys y backend de ejecución desde la UI.
3. El código generado corre en un sandbox aislado, preferiblemente dentro de vuestra nube o perímetro interno.
4. Todos los runs, artefactos e informes permanecen bajo vuestro control.
5. El producto puede empezar en modo piloto con Docker Compose y evolucionar después a una topología de producción sin cambiar la lógica principal.

## Qué Evitar en Ventas

Evita presentarlo como:
- "solo una app de Streamlit"
- "solo una herramienta de escritorio"
- "solo un `.exe`"

Esos framings hacen que el comprador enterprise piense en:
- poca mantenibilidad
- dudas de seguridad
- problemas de soporte
- falta de auditabilidad

## Qué Decir en Su Lugar

Usa este lenguaje:

- "appliance de analítica con IA self-hosted"
- "límite de ejecución controlado por el cliente"
- "configuración UI-first con secretos cifrados"
- "runtime aislado para código generado"
- "desplegable en Docker, Kubernetes o private cloud"

## Recomendación Inmediata de Productización

La secuencia más práctica para estar listos para ventas es:

1. mantener la app actual como pilot appliance
2. documentar bien el despliegue con Docker Compose
3. documentar bien la integración del sandbox gateway
4. preparar un diagrama de arquitectura de producción
5. más adelante separar UI y worker en servicios distintos

## Conclusión

El modelo profesional correcto de despliegue es:

- **no** un `.exe` como oferta principal
- **sí** un producto web self-hosted con ejecución en background y sandbox aislado

Ese es el modelo que un cliente enterprise entenderá, validará y aprobará con mucha más facilidad.
