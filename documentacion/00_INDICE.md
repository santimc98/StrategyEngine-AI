# StrategyEngine AI — Documentación

## Guías de despliegue e integración

| # | Documento | Descripción |
|---|-----------|-------------|
| 01 | [Despliegue con Docker](01_DESPLIEGUE_DOCKER.md) | Instalación rápida con Docker y Docker Compose. Configuración de volúmenes, puertos, health checks y solución de problemas. |
| 02 | [Despliegue en Cloud](02_DESPLIEGUE_CLOUD.md) | Guías paso a paso para Google Cloud (Cloud Run), Microsoft Azure (Container Apps) y AWS (ECS + Fargate). Persistencia y recomendaciones de producción. |
| 03 | [Integración con CRMs](03_INTEGRACION_CRM.md) | Configuración de HubSpot, Salesforce y Microsoft Dynamics 365. Obtención de credenciales, queries SOQL, y seguridad de las credenciales almacenadas. |
| 04 | [Sandbox Gateway](04_SANDBOX_GATEWAY.md) | Protocolo HTTP para integrar un sandbox de ejecución remoto propio de la empresa. Especificación de API, autenticación, modelo de sesión y checklist de integración. |
| 05 | [Configuración de Modelos LLM](05_CONFIGURACION_MODELOS.md) | Cómo seleccionar y configurar los modelos de IA por agente. Presets disponibles, costes estimados y recomendaciones por tipo de cliente. |
| 06 | [Onboarding del Cliente](06_ONBOARDING_CLIENTE.md) | Guía paso a paso para que el equipo técnico del cliente despliegue el producto y ejecute su primer análisis en 30 minutos. Checklist de verificación incluida. |

## Orden recomendado de lectura

- **Entregar al cliente el día de la venta**: `06_ONBOARDING_CLIENTE.md`
- **Referencia técnica de despliegue**: `01_DESPLIEGUE_DOCKER.md` y `02_DESPLIEGUE_CLOUD.md`
- **Conectar datos del cliente**: `03_INTEGRACION_CRM.md`
- **Ejecución en infraestructura del cliente**: `04_SANDBOX_GATEWAY.md`
- **Optimizar coste/calidad**: `05_CONFIGURACION_MODELOS.md`
