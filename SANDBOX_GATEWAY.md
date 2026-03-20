# Guia de Integracion del Sandbox Gateway

## Objetivo

Este documento explica como una empresa puede conectar su propio sandbox de ejecucion a StrategyEngine AI sin modificar el core del producto.

El producto soporta dos modos de ejecucion:

- `local`: la run se ejecuta en la misma maquina donde corre la app
- `remote`: la run se ejecuta a traves de un `sandbox gateway` HTTP alojado por la propia empresa

La configuracion recomendada para clientes enterprise es usar `remote` y desplegar ese gateway dentro de su propia nube o infraestructura interna. Asi, el coste de compute, almacenamiento, red y seguridad queda dentro del entorno del cliente.

## Modelo de producto

La app **no** se integra por separado con Google Cloud, Azure, AWS ni con ningun sandbox especifico de proveedor.

En su lugar, se integra con un unico protocolo:

- un gateway HTTP de sandbox alojado por la empresa que implemente una API pequena y estable

Esto implica:

- la UI es la misma para cualquier empresa
- el grafo y los agentes no necesitan cambios por cliente
- la empresa puede implementar ese gateway sobre Cloud Run, GKE, Batch, Kubernetes, VMs, contenedores o cualquier orquestador interno

## Que configura el cliente en la UI

La UI expone estos campos para el provider `remote`:

- `Endpoint del gateway`
- `API key`
- `Auth scheme`
- `Auth header`
- `Workspace`
- `Project`
- `Hint de infraestructura`
- `Timeout HTTP`

Esa configuracion se guarda localmente y se pasa a cada run como `sandbox_config`.

## API requerida del gateway

El gateway debe exponer los siguientes endpoints.

### 1. Health Check

`GET /health`

Objetivo:
- usado por la UI para validar que el gateway remoto es accesible

Respuesta esperada:

```json
{
  "ok": true
}
```

Cualquier respuesta `2xx` se considera saludable por el cliente actual.

### 2. Crear sesion

`POST /sessions`

Objetivo:
- crear una sesion aislada de ejecucion para una run

Cuerpo de la request:

```json
{
  "workspace_id": "analytics-prod",
  "project": "customer-project",
  "provider_hint": "gcp",
  "metadata": {
    "optional": "free-form metadata"
  }
}
```

Respuesta requerida:

```json
{
  "session_id": "sess-123"
}
```

Alternativa aceptada:
- `id` puede devolverse en lugar de `session_id`

### 3. Escribir archivo

`POST /sessions/{session_id}/files/write`

Objetivo:
- subir al sandbox archivos de entrada, scripts, manifests, contratos y artefactos auxiliares

Cuerpo de la request:

```json
{
  "path": "/home/user/run/abc/data/raw.csv",
  "content_base64": "SGVsbG8="
}
```

Respuesta esperada:

```json
{
  "ok": true
}
```

Si falla la escritura:

```json
{
  "ok": false,
  "error": "reason"
}
```

### 4. Leer archivo

`GET /sessions/{session_id}/files/read?path=/home/user/run/abc/data/output.csv`

Objetivo:
- descargar desde el sandbox los artefactos generados por la run

Respuesta requerida:

```json
{
  "content_base64": "SGVsbG8="
}
```

### 5. Ejecutar comando

`POST /sessions/{session_id}/commands/run`

Objetivo:
- ejecutar comandos shell dentro de la sesion del sandbox

Cuerpo de la request:

```json
{
  "cmd": "python /home/user/run/abc/ml_engineer/script.py",
  "timeout": 600
}
```

Respuesta requerida:

```json
{
  "stdout": "logs...",
  "stderr": "",
  "exit_code": 0
}
```

### 6. Cerrar sesion

`POST /sessions/{session_id}/close`

Objetivo:
- liberar los recursos del sandbox cuando termina la run

Respuesta esperada:

```json
{
  "ok": true
}
```

## Autenticacion

El cliente soporta autenticacion generica basada en cabeceras HTTP.

Campos de la UI:

- `API key`
- `Auth scheme`
- `Auth header`

Comportamiento:

- si `Auth scheme = Bearer` y `Auth header = Authorization`, las requests incluyen:

```text
Authorization: Bearer <api_key>
```

- si `Auth scheme = none`, la clave se envia directamente:

```text
<Auth header>: <api_key>
```

Esto se ha hecho asi a proposito para que cada empresa pueda adaptarlo a su convencion de autenticacion sin cambiar el producto.

## Modelo de sesion

Cada run debe mapearse a una sesion aislada.

Propiedades recomendadas:

- aislamiento de filesystem por sesion
- aislamiento de procesos por sesion
- limpieza explicita en `/close`
- enforcement de timeouts
- cuotas de recursos
- auditoria y logging

El producto asume que puede:

- subir archivos
- ejecutar comandos varias veces dentro de la misma sesion
- descargar artefactos
- cerrar la sesion cuando termina

## Semantica de archivos

La app escribe archivos usando rutas de sandbox como:

- `/home/user/run/<run_id>/...`

Tu gateway no necesita conservar esa ruta literalmente en disco, pero si debe conservar su identidad logica dentro de la sesion.

En la practica:

- mapea la ruta entrante a tu propio layout interno
- mantén estable el contrato de rutas dentro de la sesion

## Requisitos de manejo de errores

Para que el sistema funcione bien, el gateway deberia:

- devolver respuestas JSON deterministas
- preservar `stdout`, `stderr` y `exit_code`
- devolver `non-2xx` o `ok=false` estructurado cuando falle una operacion
- aplicar timeouts del lado servidor
- evitar truncar inesperadamente la salida de comandos

Recomendado:

- incluir request IDs en los logs del gateway
- incluir session IDs en todos los logs internos
- conservar logs de ejecucion para auditoria

## Recomendaciones de seguridad

Controles minimos recomendados:

- TLS habilitado
- credenciales de vida corta si es posible
- allowlisting de red si el gateway es privado
- aislamiento estricto por sesion
- no persistir credenciales dentro de las sesiones
- politica de retencion de artefactos
- limpieza explicita de sesiones

Muy recomendable:

- desplegar el gateway en la cuenta cloud del cliente
- ejecutar los workers reales del sandbox dentro del mismo boundary de seguridad
- no exponer directamente APIs de VM, Kubernetes o infraestructura al producto

## Despliegue de referencia en Google Cloud

Si el cliente usa Google Cloud, el patron recomendado es:

1. desplegar un servicio HTTP pequeno que haga de gateway
2. autenticar las requests de StrategyEngine AI hacia ese gateway
3. dejar que el gateway cree y gestione el backend real de ejecucion

Backends posibles detras del gateway:

- Cloud Run Jobs
- GKE
- Batch
- Compute Engine
- un servicio interno de sandbox que ya opere el cliente

Punto de partida recomendado en GCP:

- gateway en Cloud Run
- backend de ejecucion en Cloud Run Jobs o GKE
- logs en Cloud Logging
- artefactos en Cloud Storage si hace falta

Con ese modelo:

- la empresa paga el compute y el storage en su propia cuenta GCP
- StrategyEngine AI solo consume el gateway del cliente

## Ejemplo minimo de gateway

El gateway puede implementarse en cualquier lenguaje. El unico requisito es respetar el contrato de API anterior.

Flujo de alto nivel:

1. recibir `/sessions`
2. crear un contexto aislado de ejecucion
3. guardar metadata de sesion
4. aceptar subidas de archivos
5. ejecutar comandos
6. exponer archivos generados
7. limpiar recursos en `/close`

## Garantias del producto

Con este diseño:

- las empresas pequenas pueden usar `local`
- los clientes enterprise pueden usar `remote`
- no hacen falta cambios por empresa dentro del grafo
- no hay lock-in del producto con un proveedor cloud concreto
- el coste del sandbox queda en la infraestructura del cliente

## Expectativas actuales del cliente remoto

El cliente remoto actual espera:

- cuerpos request/response en JSON
- contenido de archivos codificado en base64
- endpoints de ciclo de vida de sesion exactamente como se describen arriba
- ejecucion de comandos que devuelva `stdout`, `stderr` y `exit_code`

Si una empresa ya tiene un sandbox interno, tiene dos opciones:

- hacer que ese sandbox exponga directamente este protocolo
- construir un adaptador fino delante de su sandbox actual

## Checklist de integracion

Antes de activar `remote` en produccion, verifica:

1. `/health` responde `2xx`
2. `/sessions` devuelve `session_id`
3. la subida de archivos funciona con payloads base64
4. la ejecucion de comandos devuelve `stdout/stderr/exit_code` completos
5. la descarga de archivos devuelve base64 valido
6. `/close` libera recursos
7. el aislamiento por sesion esta garantizado
8. la politica de timeouts esta aplicada
9. los logs pueden trazarse por `session_id`
10. endpoint y credenciales funcionan desde el host donde corre StrategyEngine AI

## Resumen

La empresa no necesita una integracion a medida dentro de la app.

Solo necesita:

- un `sandbox gateway` que siga este protocolo
- endpoint y credenciales configurados desde la UI

Una vez existe eso, el mismo producto puede funcionar:

- en local para clientes pequenos
- en remoto dentro de la nube o infraestructura propia del cliente

