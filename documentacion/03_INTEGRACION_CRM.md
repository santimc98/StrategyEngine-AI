# Guía de Integración con CRMs

## Visión general

StrategyEngine AI se conecta directamente a los CRMs más utilizados para importar datos sin necesidad de exportar archivos manualmente. Las credenciales se almacenan cifradas localmente con Fernet (AES-128-CBC).

## CRMs soportados

| CRM | Autenticación | Objetos soportados |
|-----|--------------|-------------------|
| HubSpot | Private App Token u OAuth | Contactos, Empresas, Deals, Tickets, objetos custom |
| Salesforce | Username/Password+Token u OAuth | Todos los objetos queryables via SOQL |
| Microsoft Dynamics 365 | Azure AD Client Credentials | Todas las entidades via OData Web API v9.2 |

## HubSpot

### Requisitos

1. Una cuenta de HubSpot con permisos de administrador
2. Un Private App Token o un Access Token OAuth

### Obtener el token

1. Ve a **Settings > Integrations > Private Apps** en tu cuenta de HubSpot
2. Crea una nueva Private App
3. Asigna los scopes necesarios:
   - `crm.objects.contacts.read`
   - `crm.objects.companies.read`
   - `crm.objects.deals.read`
   - `crm.objects.custom.read` (si usas objetos custom)
4. Copia el token generado

### Configurar en StrategyEngine AI

1. En la interfaz web, ve al panel lateral
2. Selecciona **Conectar CRM > HubSpot**
3. Pega el token en el campo **API Token**
4. Haz clic en **Probar conexión** para verificar
5. Selecciona el objeto que quieres analizar (Contacts, Companies, Deals, etc.)
6. Configura el límite de filas si es necesario
7. Haz clic en **Importar datos**

### Notas técnicas

- La paginación es automática (cursor-based) para datasets grandes
- El conector descubre automáticamente las propiedades disponibles del objeto
- Se respetan los rate-limits de la API de HubSpot (100 requests/10s)

---

## Salesforce

### Requisitos

1. Una cuenta de Salesforce con permisos de API
2. Credenciales de autenticación (una de las dos opciones):
   - **Opción A**: Username + Password + Security Token
   - **Opción B**: OAuth Access Token

### Obtener el Security Token (Opción A)

1. En Salesforce, ve a **Settings > My Personal Information > Reset My Security Token**
2. Recibirás el token por email
3. El password para la conexión será: `tu_password` + `security_token` concatenados

### Configurar en StrategyEngine AI

1. En la interfaz web, selecciona **Conectar CRM > Salesforce**
2. Introduce:
   - **Username**: tu email de Salesforce
   - **Password**: tu password + security token
   - **Dominio**: `login.salesforce.com` (producción) o `test.salesforce.com` (sandbox)
3. Haz clic en **Probar conexión**
4. Escribe una query SOQL o selecciona un objeto predefinido
5. Haz clic en **Importar datos**

### Queries SOQL

Puedes escribir queries SOQL directas para seleccionar exactamente los datos que necesitas:

```sql
SELECT Id, Name, Amount, StageName, CloseDate
FROM Opportunity
WHERE StageName != 'Closed Won'
AND CreatedDate >= 2024-01-01T00:00:00Z
```

### Notas técnicas

- Gestión automática de sesiones y reconexión
- Detección de rate-limits con backoff automático
- Soporte para todos los objetos queryables de Salesforce

---

## Microsoft Dynamics 365

### Requisitos

1. Un tenant de Azure con Dynamics 365
2. Una App Registration en Azure AD con permisos de Dynamics

### Configurar Azure AD

1. Ve a **Azure Portal > Azure Active Directory > App Registrations**
2. Crea una nueva registration
3. En **API Permissions**, añade:
   - `Dynamics CRM > user_impersonation`
4. En **Certificates & Secrets**, crea un Client Secret
5. Anota:
   - **Client ID** (Application ID)
   - **Client Secret**
   - **Tenant ID**
   - **Resource URL** de tu instancia Dynamics (ej: `https://tuorg.crm4.dynamics.com`)

### Configurar en StrategyEngine AI

1. En la interfaz web, selecciona **Conectar CRM > Dynamics 365**
2. Introduce:
   - **Client ID**
   - **Client Secret**
   - **Tenant ID**
   - **Resource URL**
3. Haz clic en **Probar conexión**
4. Selecciona la entidad a importar (Accounts, Contacts, Leads, etc.)
5. Haz clic en **Importar datos**

### Notas técnicas

- Integración via OData Web API v9.2
- Soporte multi-tenant (workspace_id aísla por organización)
- Paginación automática para datasets grandes

---

## Seguridad de las credenciales

### Cómo se almacenan

- Las credenciales CRM **nunca** se envían a APIs externas ni a los LLMs
- Se almacenan cifradas localmente con Fernet (clave derivada de la máquina)
- En la interfaz solo se muestran los primeros 4 y últimos 4 caracteres
- El archivo cifrado (`data/api_keys.enc`) debe incluirse en los backups

### Recomendaciones

- Usa tokens de vida corta cuando sea posible (OAuth refresh tokens)
- Rota las credenciales periódicamente
- En despliegues enterprise, considera usar un gestor de secretos externo (Vault, Secret Manager) y pasar las credenciales como variables de entorno

---

## Alternativa: carga manual de CSV

Si el CRM no está soportado o la empresa prefiere no dar acceso directo:

1. Exporta los datos desde tu CRM en formato CSV
2. Sube el archivo directamente desde la interfaz web de StrategyEngine AI
3. El sistema detecta automáticamente el encoding, delimitador y estructura

Formatos soportados: CSV, Excel (.xlsx, .xls)

Tamaño máximo de upload: 1 GB (configurable en `.streamlit/config.toml`)
