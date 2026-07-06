"""Runtime SSL compatibility patch for local Windows conda environments.

Some Windows certificate stores contain malformed certificate data. Libraries
such as Tornado/Streamlit may call ``ssl.create_default_context()`` during
import, causing Python to fail before app code starts. Python imports this file
automatically at startup when it is on ``sys.path``; the patch retries with
certifi's CA bundle only when the default context creation hits the known ASN1
certificate-store error.
"""

from __future__ import annotations

import ssl
from typing import Optional, Union


_original_create_default_context = ssl.create_default_context


def _patched_create_default_context(
    purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
    *,
    cafile: Optional[str] = None,
    capath: Optional[str] = None,
    cadata: Optional[Union[str, bytes]] = None,
) -> ssl.SSLContext:
    """Create an SSL context, falling back to certifi for malformed stores."""

    try:
        return _original_create_default_context(
            purpose=purpose,
            cafile=cafile,
            capath=capath,
            cadata=cadata,
        )
    except ssl.SSLError as exc:
        if "ASN1" not in str(exc) and "NOT_ENOUGH_DATA" not in str(exc):
            raise
        try:
            import certifi
        except Exception:
            raise exc
        return _original_create_default_context(
            purpose=purpose,
            cafile=certifi.where(),
            capath=capath,
            cadata=cadata,
        )


ssl.create_default_context = _patched_create_default_context
