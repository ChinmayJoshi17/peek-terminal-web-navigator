"""UI package — rich terminal presentation, completely decoupled from business logic.

Per the user's instruction to restructure nano-browser-agent into a modular,
production-ready system architecture per ECC specs, this package provides
the presentation layer that is completely decoupled from business logic.

Importers/callers: main.py imports from ui.console for rendering loops.
Affected API: None (pure presentation layer).
Data schemas: None (consumes types from core.types but does not define schemas).
"""