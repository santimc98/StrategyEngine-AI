"""Microsoft Dynamics 365 CRM connector using MSAL + OData Web API."""

from __future__ import annotations

import pandas as pd
import requests
import msal

from src.connectors.base import CRMConnector, CRMAuthError, CRMConnectionError, CRMRateLimitError

_API_VERSION = "v9.2"
_PAGE_SIZE = 5000  # Dynamics hard max per page


class DynamicsConnector(CRMConnector):
    """Connect to Dynamics 365 CRM via client credentials (Azure AD app)."""

    def __init__(self) -> None:
        self.access_token: str | None = None
        self.base_url: str | None = None
        self._session: requests.Session | None = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def authenticate(self, credentials: dict) -> bool:
        """Authenticate with Dynamics 365 using client credentials.

        Required keys in *credentials*:
        * ``tenant_id`` – Azure AD tenant.
        * ``client_id`` – App registration client ID.
        * ``client_secret`` – App registration secret.
        * ``crm_url`` – Organization URL, e.g. ``https://myorg.crm.dynamics.com``.
        """
        tenant_id = credentials.get("tenant_id", "").strip()
        client_id = credentials.get("client_id", "").strip()
        client_secret = credentials.get("client_secret", "").strip()
        crm_url = credentials.get("crm_url", "").strip().rstrip("/")

        if not all([tenant_id, client_id, client_secret, crm_url]):
            raise CRMAuthError("Faltan credenciales: tenant_id, client_id, client_secret y crm_url son obligatorios.")

        authority = f"https://login.microsoftonline.com/{tenant_id}"
        scope = [f"{crm_url}/.default"]

        try:
            app = msal.ConfidentialClientApplication(
                client_id=client_id,
                client_credential=client_secret,
                authority=authority,
            )
            result = app.acquire_token_for_client(scopes=scope)
        except Exception as exc:
            raise CRMConnectionError(f"Error al conectar con Azure AD: {exc}") from exc

        if "access_token" not in result:
            error_desc = result.get("error_description", result.get("error", "Unknown"))
            raise CRMAuthError(f"Autenticacion fallida: {error_desc}")

        self.access_token = result["access_token"]
        self.base_url = f"{crm_url}/api/data/{_API_VERSION}"
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0",
        })

        # Validate the token actually works
        try:
            resp = self._session.get(f"{self.base_url}/WhoAmI()", timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            self.access_token = None
            self._session = None
            raise CRMAuthError(f"Token valido pero no se pudo verificar acceso al CRM: {exc}") from exc

        return True

    # ------------------------------------------------------------------
    def test_connection(self) -> bool:
        if self._session is None or self.base_url is None:
            return False
        try:
            resp = self._session.get(f"{self.base_url}/WhoAmI()", timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Object exploration
    # ------------------------------------------------------------------
    def list_objects(self) -> list[dict]:
        if self._session is None or self.base_url is None:
            raise CRMConnectionError("No autenticado. Llama a authenticate() primero.")

        url = (
            f"{self.base_url}/EntityDefinitions"
            "?$select=LogicalName,DisplayName,EntitySetName"
            "&$filter=IsValidForAdvancedFind eq true"
        )
        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            raise CRMConnectionError(f"Error al listar entidades: {exc}") from exc

        entities = resp.json().get("value", [])
        objects: list[dict] = []
        for entity in entities:
            entity_set = entity.get("EntitySetName")
            if not entity_set:
                continue
            display = entity.get("DisplayName") or {}
            localized = display.get("UserLocalizedLabel") or {}
            label = localized.get("Label") or entity.get("LogicalName", entity_set)
            objects.append({"name": entity_set, "label": label})

        objects.sort(key=lambda o: o["label"])
        return objects

    # ------------------------------------------------------------------
    # Data fetch
    # ------------------------------------------------------------------
    def fetch_object_data(self, object_name: str, max_records: int = 10000) -> pd.DataFrame:
        if self._session is None or self.base_url is None:
            raise CRMConnectionError("No autenticado. Llama a authenticate() primero.")

        page_size = min(_PAGE_SIZE, max_records)
        url: str | None = f"{self.base_url}/{object_name}?$top={page_size}&$count=true"
        all_records: list[dict] = []

        while url and len(all_records) < max_records:
            try:
                resp = self._session.get(url, timeout=60)
            except Exception as exc:
                raise CRMConnectionError(f"Error al obtener datos de '{object_name}': {exc}") from exc

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "60")
                raise CRMRateLimitError(
                    f"Limite de requests excedido. Reintenta en {retry_after}s."
                )
            if resp.status_code == 401:
                raise CRMAuthError("Token expirado o invalido. Reconecta.")
            if resp.status_code == 404:
                raise CRMConnectionError(
                    f"Entidad '{object_name}' no encontrada. Verifica el nombre."
                )
            resp.raise_for_status()

            data = resp.json()
            records = data.get("value", [])
            all_records.extend(records)
            url = data.get("@odata.nextLink")

        # Trim to max_records
        all_records = all_records[:max_records]

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        # Strip OData annotation columns
        clean_cols = [c for c in df.columns if not c.startswith(("@", "_"))]
        df = df[clean_cols]

        return df
