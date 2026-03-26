# Guía de Despliegue en Cloud

## Opciones de despliegue

StrategyEngine AI puede desplegarse en cualquier proveedor cloud que soporte contenedores Docker. Esta guía cubre los tres principales.

## Google Cloud (Cloud Run)

### Requisitos

- Cuenta de Google Cloud con billing habilitado
- `gcloud` CLI instalado y autenticado
- Docker instalado localmente

### Pasos

#### 1. Crear proyecto y habilitar APIs

```bash
gcloud projects create strategyengine-prod --name="StrategyEngine AI"
gcloud config set project strategyengine-prod
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

#### 2. Crear repositorio de imágenes

```bash
gcloud artifacts repositories create strategyengine \
  --repository-format=docker \
  --location=europe-west1
```

#### 3. Construir y subir la imagen

```bash
gcloud auth configure-docker europe-west1-docker.pkg.dev

docker build -t europe-west1-docker.pkg.dev/strategyengine-prod/strategyengine/app:latest .

docker push europe-west1-docker.pkg.dev/strategyengine-prod/strategyengine/app:latest
```

#### 4. Desplegar en Cloud Run

```bash
gcloud run deploy strategyengine-ai \
  --image=europe-west1-docker.pkg.dev/strategyengine-prod/strategyengine/app:latest \
  --region=europe-west1 \
  --port=8501 \
  --memory=8Gi \
  --cpu=4 \
  --timeout=3600 \
  --set-env-vars="OPENROUTER_API_KEY=sk-or-v1-tu-clave" \
  --allow-unauthenticated
```

Notas:
- `--memory=8Gi` es el mínimo recomendado para las dependencias ML
- `--timeout=3600` permite runs de hasta 1 hora
- Para restringir el acceso, quita `--allow-unauthenticated` y configura IAM

#### 5. Configurar dominio personalizado (opcional)

```bash
gcloud run domain-mappings create \
  --service=strategyengine-ai \
  --domain=app.tudominio.com \
  --region=europe-west1
```

### Persistencia en GCP

Cloud Run es stateless. Para persistir runs y datos:

- Monta un bucket de Cloud Storage como volumen
- O usa Filestore (NFS) para persistencia compartida

```bash
gcloud run services update strategyengine-ai \
  --add-volume=name=runs-vol,type=cloud-storage,bucket=strategyengine-runs \
  --add-volume-mount=volume=runs-vol,mount-path=/app/runs
```

---

## Microsoft Azure (Container Apps)

### Requisitos

- Cuenta de Azure con suscripción activa
- `az` CLI instalado y autenticado

### Pasos

#### 1. Crear grupo de recursos

```bash
az group create --name strategyengine-rg --location westeurope
```

#### 2. Crear registro de contenedores

```bash
az acr create --name strategyengineacr --resource-group strategyengine-rg --sku Basic
az acr login --name strategyengineacr
```

#### 3. Construir y subir la imagen

```bash
docker build -t strategyengineacr.azurecr.io/app:latest .
docker push strategyengineacr.azurecr.io/app:latest
```

#### 4. Crear entorno de Container Apps

```bash
az containerapp env create \
  --name strategyengine-env \
  --resource-group strategyengine-rg \
  --location westeurope
```

#### 5. Desplegar

```bash
az containerapp create \
  --name strategyengine-ai \
  --resource-group strategyengine-rg \
  --environment strategyengine-env \
  --image strategyengineacr.azurecr.io/app:latest \
  --target-port 8501 \
  --ingress external \
  --cpu 4 \
  --memory 8Gi \
  --env-vars "OPENROUTER_API_KEY=sk-or-v1-tu-clave" \
  --registry-server strategyengineacr.azurecr.io
```

### Persistencia en Azure

Usa Azure Files para montar almacenamiento persistente en Container Apps:

```bash
az storage account create --name strategyenginestor --resource-group strategyengine-rg
az storage share create --name runs --account-name strategyenginestor
```

---

## Amazon Web Services (ECS + Fargate)

### Requisitos

- Cuenta de AWS con permisos de ECS
- `aws` CLI instalado y autenticado

### Pasos

#### 1. Crear repositorio ECR

```bash
aws ecr create-repository --repository-name strategyengine-ai --region eu-west-1
```

#### 2. Construir y subir la imagen

```bash
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-west-1.amazonaws.com

docker build -t <account-id>.dkr.ecr.eu-west-1.amazonaws.com/strategyengine-ai:latest .

docker push <account-id>.dkr.ecr.eu-west-1.amazonaws.com/strategyengine-ai:latest
```

#### 3. Crear task definition

Crea un archivo `task-definition.json`:

```json
{
  "family": "strategyengine-ai",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "app",
      "image": "<account-id>.dkr.ecr.eu-west-1.amazonaws.com/strategyengine-ai:latest",
      "portMappings": [{"containerPort": 8501}],
      "environment": [
        {"name": "OPENROUTER_API_KEY", "value": "sk-or-v1-tu-clave"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/strategyengine-ai",
          "awslogs-region": "eu-west-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### 4. Crear servicio

```bash
aws ecs create-cluster --cluster-name strategyengine

aws ecs register-task-definition --cli-input-json file://task-definition.json

aws ecs create-service \
  --cluster strategyengine \
  --service-name strategyengine-ai \
  --task-definition strategyengine-ai \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Persistencia en AWS

Usa EFS (Elastic File System) montado como volumen en la task definition para persistir `/app/runs` y `/app/data`.

---

## Recomendaciones generales para producción

### Seguridad

- Nunca expongas la aplicación sin autenticación en producción
- Usa un reverse proxy (Nginx, Traefik, Cloud Load Balancer) con HTTPS
- Almacena las API keys en un gestor de secretos (Secret Manager, Key Vault, Secrets Manager) en vez de variables de entorno planas
- Restringe el acceso por IP o VPN si es un despliegue interno

### Rendimiento

- Mínimo 8 GB de RAM para el contenedor
- 4 CPUs recomendadas para ejecución de ML
- SSD para los volúmenes de datos (mejora I/O en datasets grandes)

### Monitorización

- Configura alertas en el health check (`/_stcore/health`)
- Monitoriza el uso de memoria del contenedor (las runs ML pueden ser intensivas)
- Revisa los logs periódicamente para detectar errores de API key agotada

### Backups

- Programa backups regulares del volumen `runs/` (contiene todo el historial de ejecuciones)
- El volumen `data/` contiene credenciales CRM cifradas — inclúyelo en el backup
