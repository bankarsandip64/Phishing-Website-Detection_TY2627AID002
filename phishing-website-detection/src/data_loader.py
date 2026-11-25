"""Dataset loading, caching, and feature metadata for Phishing Website Detection."""

from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from scipy.io import arff

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_PATH = DATA_DIR / "phishing_websites.csv"

UCI_ARFF_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
)
OPENML_ARFF_GZ_URL = (
    "https://www.openml.org/data/v1/download/1798106/PhishingWebsites.arff.gz"
)

TARGET_COLUMN = "Result"

# Feature groups and human-readable descriptions for UI and documentation
FEATURE_METADATA: Dict[str, Dict[str, object]] = {
    "having_IP_Address": {
        "group": "URL / Address Bar",
        "label": "IP Address in URL",
        "description": "Whether the URL uses an IP address instead of a domain name (-1: Yes/Phishing, 1: No/Legitimate).",
        "values": [-1, 1],
    },
    "URL_Length": {
        "group": "URL / Address Bar",
        "label": "URL Length",
        "description": "Length of the URL (-1: >=75 chars, 0: 54-74 chars, 1: <54 chars).",
        "values": [-1, 0, 1],
    },
    "Shortining_Service": {
        "group": "URL / Address Bar",
        "label": "URL Shortening Service",
        "description": "Whether a URL shortener (e.g., bit.ly, tinyurl) is used (-1: Yes, 1: No).",
        "values": [-1, 1],
    },
    "having_At_Symbol": {
        "group": "URL / Address Bar",
        "label": "'@' Symbol in URL",
        "description": "Presence of '@' symbol which can obscure the real destination (-1: Yes, 1: No).",
        "values": [-1, 1],
    },
    "double_slash_redirecting": {
        "group": "URL / Address Bar",
        "label": "Double-Slash Redirecting",
        "description": "Position of '//' beyond protocol prefix (-1: Suspicious, 1: Normal).",
        "values": [-1, 1],
    },
    "Prefix_Suffix": {
        "group": "URL / Address Bar",
        "label": "Prefix/Suffix Hyphen in Domain",
        "description": "Presence of '-' in domain name to mimic legitimate brands (-1: Yes, 1: No).",
        "values": [-1, 1],
    },
    "having_Sub_Domain": {
        "group": "URL / Address Bar",
        "label": "Sub-Domain Count",
        "description": "Number of dots in domain (-1: >2 dots, 0: 2 dots, 1: <=1 dot).",
        "values": [-1, 0, 1],
    },
    "SSLfinal_State": {
        "group": "URL / Address Bar",
        "label": "SSL / HTTPS Certificate State",
        "description": "HTTPS presence, certificate issuer trust, and certificate age (-1: Untrusted/None, 0: Suspicious, 1: Trusted).",
        "values": [-1, 0, 1],
    },
    "Domain_registeration_length": {
        "group": "Domain Information",
        "label": "Domain Registration Length",
        "description": "Time until domain expiration (-1: <=1 year, 1: >1 year).",
        "values": [-1, 1],
    },
    "Favicon": {
        "group": "Webpage Features",
        "label": "External Favicon",
        "description": "Whether the favicon is loaded from an external domain (-1: External, 1: Same domain).",
        "values": [-1, 1],
    },
    "port": {
        "group": "URL / Address Bar",
        "label": "Non-Standard Port",
        "description": "Whether non-standard ports are open/used (-1: Non-standard, 1: Standard).",
        "values": [-1, 1],
    },
    "HTTPS_token": {
        "group": "URL / Address Bar",
        "label": "'https' Token in Domain",
        "description": "Deceptive 'https' string inside the domain part of URL (-1: Yes, 1: No).",
        "values": [-1, 1],
    },
    "Request_URL": {
        "group": "Webpage Features",
        "label": "External Request URLs",
        "description": "Percentage of page objects (images/videos) loaded from external domains (-1: >61%, 0: 22-61%, 1: <22%).",
        "values": [-1, 0, 1],
    },
    "URL_of_Anchor": {
        "group": "Webpage Features",
        "label": "Suspicious Anchor URLs",
        "description": "Percentage of <a> tags pointing to external or void targets (-1: >67%, 0: 31-67%, 1: <31%).",
        "values": [-1, 0, 1],
    },
    "Links_in_tags": {
        "group": "Webpage Features",
        "label": "External Links in Meta/Script/Link Tags",
        "description": "Percentage of links in <Meta>, <Script>, <Link> tags (-1: >81%, 0: 17-81%, 1: <17%).",
        "values": [-1, 0, 1],
    },
    "SFH": {
        "group": "Webpage Features",
        "label": "Server Form Handler (SFH)",
        "description": "Where submitted form data goes (-1: Empty/about:blank, 0: External domain, 1: Same domain).",
        "values": [-1, 0, 1],
    },
    "Submitting_to_email": {
        "group": "Webpage Features",
        "label": "Submitting Form to Email",
        "description": "Whether form uses mailto: or PHP mail() (-1: Yes, 1: No).",
        "values": [-1, 1],
    },
    "Abnormal_URL": {
        "group": "Webpage Features",
        "label": "Abnormal URL (WHOIS Mismatch)",
        "description": "Whether hostname is missing from WHOIS record (-1: Abnormal, 1: Normal).",
        "values": [-1, 1],
    },
    "Redirect": {
        "group": "Webpage Features",
        "label": "Website Forwarding / Redirects",
        "description": "Number of redirects before landing page (0: >=4 redirects, 1: <=1 redirect).",
        "values": [0, 1],
    },
    "on_mouseover": {
        "group": "Webpage Features",
        "label": "Status Bar Manipulation (onMouseOver)",
        "description": "JavaScript altering the status bar on hover (-1: Yes, 1: No).",
        "values": [-1, 1],
    },
    "RightClick": {
        "group": "Webpage Features",
        "label": "Right-Click Disabled",
        "description": "Whether right-click context menu is disabled via JS (-1: Disabled, 1: Enabled).",
        "values": [-1, 1],
    },
    "popUpWidnow": {
        "group": "Webpage Features",
        "label": "Pop-up Window Prompt",
        "description": "Presence of pop-up windows asking for credentials (-1: Yes, 1: No).",
        "values": [-1, 1],
    },
    "Iframe": {
        "group": "Webpage Features",
        "label": "Hidden Iframe Redirection",
        "description": "Use of invisible <iframe> tags (-1: Yes, 1: No).",
        "values": [-1, 1],
    },
    "age_of_domain": {
        "group": "Domain Information",
        "label": "Age of Domain",
        "description": "Domain age from WHOIS (-1: <6 months, 1: >=6 months).",
        "values": [-1, 1],
    },
    "DNSRecord": {
        "group": "Domain Information",
        "label": "DNS Record Availability",
        "description": "Whether a valid DNS record exists (-1: Missing, 1: Present).",
        "values": [-1, 1],
    },
    "web_traffic": {
        "group": "Domain Information",
        "label": "Web Traffic Rank",
        "description": "Site popularity/traffic rank (-1: No rank/very low, 0: >100k, 1: <=100k).",
        "values": [-1, 0, 1],
    },
    "Page_Rank": {
        "group": "Domain Information",
        "label": "PageRank Score",
        "description": "PageRank authority (-1: <0.2, 1: >=0.2).",
        "values": [-1, 1],
    },
    "Google_Index": {
        "group": "Domain Information",
        "label": "Indexed by Google",
        "description": "Whether the webpage is indexed by Google (-1: No, 1: Yes).",
        "values": [-1, 1],
    },
    "Links_pointing_to_page": {
        "group": "Domain Information",
        "label": "Inbound Links Count",
        "description": "Number of external links pointing to the page (-1: 0 links, 0: 1-2 links, 1: >2 links).",
        "values": [-1, 0, 1],
    },
    "Statistical_report": {
        "group": "Domain Information",
        "label": "Phishing Blacklist / Statistical Report",
        "description": "Whether host/IP appears in PhishTank/StopBadware reports (-1: Listed, 1: Clean).",
        "values": [-1, 1],
    },
}


def _parse_arff_stream(text_stream: io.TextIOBase) -> pd.DataFrame:
    """Parse an ARFF text stream into an integer DataFrame."""
    data, _ = arff.loadarff(text_stream)
    df = pd.DataFrame(data)
    return df.map(lambda x: int(x.decode("utf-8")) if isinstance(x, bytes) else int(x))


def download_dataset(cache_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Download the UCI Phishing Websites dataset and cache it as CSV."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for url in (UCI_ARFF_URL, OPENML_ARFF_GZ_URL):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                raw_bytes = response.read()
            if url.endswith(".gz"):
                with gzip.open(io.BytesIO(raw_bytes), "rt", encoding="utf-8") as gz_file:
                    df = _parse_arff_stream(gz_file)
            else:
                text_stream = io.StringIO(raw_bytes.decode("utf-8"))
                df = _parse_arff_stream(text_stream)
            df.to_csv(cache_path, index=False)
            return df
        except Exception as exc:  # pragma: no cover - network fallback
            last_error = exc

    raise RuntimeError(
        f"Failed to download dataset from UCI or OpenML: {last_error}"
    )


def load_dataset(
    cache_path: Path = DEFAULT_DATA_PATH,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load the Phishing Websites dataset from local CSV cache, downloading if needed."""
    if cache_path.exists() and not force_download:
        return pd.read_csv(cache_path)
    return download_dataset(cache_path=cache_path)


def get_feature_names() -> List[str]:
    """Return ordered list of the 30 input feature names."""
    return list(FEATURE_METADATA.keys())


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Compute basic dataset summary statistics for documentation and UI."""
    phishing_count = int((df[TARGET_COLUMN] == -1).sum())
    legit_count = int((df[TARGET_COLUMN] == 1).sum())
    return {
        "n_rows": int(len(df)),
        "n_features": int(df.shape[1] - 1),
        "phishing_count": phishing_count,
        "legitimate_count": legit_count,
        "phishing_rate": round(phishing_count / len(df), 4),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
