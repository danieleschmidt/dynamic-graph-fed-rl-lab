"""Internationalization (i18n) manager for global federated RL systems."""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import threading


class LocaleCategory(Enum):
    """Categories of localizable content."""
    UI_TEXT = "ui_text"
    ERROR_MESSAGES = "error_messages"
    LOGGING = "logging"
    METRICS = "metrics"
    DOCUMENTATION = "documentation"
    API_RESPONSES = "api_responses"


@dataclass
class LocaleConfig:
    """Configuration for a specific locale."""
    
    code: str  # e.g., "en-US", "fr-FR", "zh-CN"
    name: str  # e.g., "English (United States)"
    language: str  # e.g., "English"
    country: str  # e.g., "United States"
    rtl: bool = False  # Right-to-left text direction
    currency: str = "USD"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "1,234.56"
    encoding: str = "utf-8"
    fallback_locale: Optional[str] = None


class InternationalizationManager:
    """
    Manages internationalization for global federated RL systems.
    
    Features:
    - Multi-language support for UI and API responses
    - Locale-specific formatting (dates, numbers, currency)
    - Dynamic language switching
    - Translation key management
    - Fallback language handling
    - Cultural adaptation for different regions
    """
    
    def __init__(self, default_locale: str = "en-US"):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.locales: Dict[str, LocaleConfig] = {}
        self.translations: Dict[str, Dict[str, str]] = {}  # locale -> {key: translation}
        self.translation_cache: Dict[str, str] = {}
        
        # Thread safety
        self.locale_lock = threading.RLock()
        
        # Translation providers and formatters
        self.formatters: Dict[str, Callable] = {}
        self.translation_providers: List[Callable] = []
        
        # Metrics
        self.translation_metrics = {
            "requests": 0,
            "cache_hits": 0,
            "fallbacks_used": 0,
            "missing_translations": set()
        }
        
        print("🌐 Internationalization manager initialized")
        
        # Setup default locales
        self._setup_default_locales()
        self._setup_core_translations()
    
    def _setup_default_locales(self):
        """Setup default supported locales."""
        default_locales = [
            LocaleConfig(
                code="en-US",
                name="English (United States)",
                language="English",
                country="United States",
                currency="USD",
                date_format="%m/%d/%Y",
                time_format="%I:%M:%S %p"
            ),
            LocaleConfig(
                code="en-GB", 
                name="English (United Kingdom)",
                language="English",
                country="United Kingdom",
                currency="GBP",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                fallback_locale="en-US"
            ),
            LocaleConfig(
                code="fr-FR",
                name="Français (France)",
                language="French",
                country="France",
                currency="EUR",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                number_format="1 234,56",
                fallback_locale="en-US"
            ),
            LocaleConfig(
                code="de-DE",
                name="Deutsch (Deutschland)",
                language="German", 
                country="Germany",
                currency="EUR",
                date_format="%d.%m.%Y",
                time_format="%H:%M:%S",
                number_format="1.234,56",
                fallback_locale="en-US"
            ),
            LocaleConfig(
                code="es-ES",
                name="Español (España)",
                language="Spanish",
                country="Spain", 
                currency="EUR",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                number_format="1.234,56",
                fallback_locale="en-US"
            ),
            LocaleConfig(
                code="zh-CN",
                name="中文 (中国)",
                language="Chinese",
                country="China",
                currency="CNY",
                date_format="%Y年%m月%d日",
                time_format="%H:%M:%S",
                fallback_locale="en-US"
            ),
            LocaleConfig(
                code="ja-JP",
                name="日本語 (日本)",
                language="Japanese",
                country="Japan",
                currency="JPY",
                date_format="%Y年%m月%d日",
                time_format="%H:%M:%S",
                fallback_locale="en-US"
            ),
            LocaleConfig(
                code="ar-SA",
                name="العربية (المملكة العربية السعودية)",
                language="Arabic",
                country="Saudi Arabia",
                rtl=True,
                currency="SAR",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                fallback_locale="en-US"
            )
        ]
        
        for locale in default_locales:
            self.register_locale(locale)
    
    def _setup_core_translations(self):
        """Setup core system translations."""
        core_translations = {
            "en-US": {
                "system.startup": "System starting up",
                "system.shutdown": "System shutting down", 
                "agent.deployed": "Agent deployed successfully",
                "agent.failed": "Agent deployment failed",
                "federation.sync": "Federation synchronization complete",
                "quantum.task.created": "Quantum task created",
                "error.connection_failed": "Connection failed",
                "error.invalid_input": "Invalid input provided",
                "error.timeout": "Operation timed out",
                "metric.performance": "Performance",
                "metric.latency": "Latency",
                "metric.throughput": "Throughput",
                "region.healthy": "Region is healthy",
                "region.degraded": "Region is degraded",
                "region.offline": "Region is offline"
            },
            "fr-FR": {
                "system.startup": "Démarrage du système",
                "system.shutdown": "Arrêt du système",
                "agent.deployed": "Agent déployé avec succès",
                "agent.failed": "Échec du déploiement de l'agent",
                "federation.sync": "Synchronisation de la fédération terminée",
                "quantum.task.created": "Tâche quantique créée",
                "error.connection_failed": "Échec de la connexion",
                "error.invalid_input": "Entrée invalide fournie",
                "error.timeout": "Délai d'attente dépassé",
                "metric.performance": "Performance",
                "metric.latency": "Latence",
                "metric.throughput": "Débit",
                "region.healthy": "La région est saine",
                "region.degraded": "La région est dégradée",
                "region.offline": "La région est hors ligne"
            },
            "de-DE": {
                "system.startup": "System startet",
                "system.shutdown": "System wird heruntergefahren",
                "agent.deployed": "Agent erfolgreich bereitgestellt",
                "agent.failed": "Agent-Bereitstellung fehlgeschlagen",
                "federation.sync": "Föderationssynchronisation abgeschlossen",
                "quantum.task.created": "Quantenaufgabe erstellt",
                "error.connection_failed": "Verbindung fehlgeschlagen",
                "error.invalid_input": "Ungültige Eingabe bereitgestellt",
                "error.timeout": "Zeitüberschreitung der Operation",
                "metric.performance": "Leistung",
                "metric.latency": "Latenz",
                "metric.throughput": "Durchsatz",
                "region.healthy": "Region ist gesund",
                "region.degraded": "Region ist beeinträchtigt",
                "region.offline": "Region ist offline"
            },
            "zh-CN": {
                "system.startup": "系统正在启动",
                "system.shutdown": "系统正在关闭",
                "agent.deployed": "代理部署成功",
                "agent.failed": "代理部署失败",
                "federation.sync": "联邦同步完成",
                "quantum.task.created": "量子任务已创建",
                "error.connection_failed": "连接失败",
                "error.invalid_input": "提供的输入无效",
                "error.timeout": "操作超时",
                "metric.performance": "性能",
                "metric.latency": "延迟",
                "metric.throughput": "吞吐量",
                "region.healthy": "区域健康",
                "region.degraded": "区域性能下降",
                "region.offline": "区域离线"
            }
        }
        
        for locale_code, translations in core_translations.items():
            if locale_code not in self.translations:
                self.translations[locale_code] = {}
            self.translations[locale_code].update(translations)
        
        print(f"🌐 Loaded translations for {len(core_translations)} locales")
    
    def register_locale(self, locale_config: LocaleConfig):
        """Register a new locale."""
        with self.locale_lock:
            self.locales[locale_config.code] = locale_config
            if locale_config.code not in self.translations:
                self.translations[locale_config.code] = {}
            
            print(f"🌐 Registered locale: {locale_config.name} ({locale_config.code})")
    
    def set_locale(self, locale_code: str) -> bool:
        """Set the current active locale."""
        with self.locale_lock:
            if locale_code in self.locales:
                self.current_locale = locale_code
                self.translation_cache.clear()  # Clear cache when locale changes
                print(f"🌐 Locale changed to: {self.locales[locale_code].name}")
                return True
            else:
                print(f"⚠️  Locale not found: {locale_code}")
                return False
    
    def translate(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Translate a key to the specified or current locale."""
        target_locale = locale or self.current_locale
        cache_key = f"{target_locale}:{key}"
        
        # Check cache first
        if cache_key in self.translation_cache:
            self.translation_metrics["cache_hits"] += 1
            translation = self.translation_cache[cache_key]
        else:
            translation = self._get_translation(key, target_locale)
            self.translation_cache[cache_key] = translation
        
        self.translation_metrics["requests"] += 1
        
        # Apply string formatting if kwargs provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                print(f"⚠️  Translation formatting error for key '{key}': {e}")
        
        return translation
    
    def _get_translation(self, key: str, locale: str) -> str:
        """Get translation for key in specified locale with fallback."""
        # Try direct translation
        if locale in self.translations and key in self.translations[locale]:
            return self.translations[locale][key]
        
        # Try fallback locale
        if locale in self.locales:
            fallback_locale = self.locales[locale].fallback_locale
            if fallback_locale and fallback_locale in self.translations:
                if key in self.translations[fallback_locale]:
                    self.translation_metrics["fallbacks_used"] += 1
                    return self.translations[fallback_locale][key]
        
        # Try default locale
        if self.default_locale in self.translations and key in self.translations[self.default_locale]:
            self.translation_metrics["fallbacks_used"] += 1
            return self.translations[self.default_locale][key]
        
        # Translation not found
        self.translation_metrics["missing_translations"].add(key)
        print(f"⚠️  Missing translation for key: {key} (locale: {locale})")
        return f"[{key}]"  # Return key in brackets as fallback
    
    def add_translations(self, locale: str, translations: Dict[str, str]):
        """Add translations for a specific locale."""
        with self.locale_lock:
            if locale not in self.translations:
                self.translations[locale] = {}
            
            self.translations[locale].update(translations)
            self.translation_cache.clear()  # Clear cache after adding translations
            
            print(f"🌐 Added {len(translations)} translations for {locale}")
    
    def format_number(self, number: float, locale: Optional[str] = None) -> str:
        """Format number according to locale conventions."""
        target_locale = locale or self.current_locale
        
        if target_locale not in self.locales:
            target_locale = self.default_locale
        
        locale_config = self.locales[target_locale]
        
        # Simple formatting based on locale patterns
        if locale_config.number_format == "1.234,56":  # European style
            return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        elif locale_config.number_format == "1 234,56":  # French style
            return f"{number:,.2f}".replace(",", " ").replace(".", ",")
        else:  # Default US style
            return f"{number:,.2f}"
    
    def format_currency(self, amount: float, locale: Optional[str] = None) -> str:
        """Format currency according to locale conventions."""
        target_locale = locale or self.current_locale
        
        if target_locale not in self.locales:
            target_locale = self.default_locale
        
        locale_config = self.locales[target_locale]
        formatted_number = self.format_number(amount, target_locale)
        
        # Currency symbol placement varies by locale
        currency = locale_config.currency
        if currency == "EUR":
            return f"{formatted_number} €"
        elif currency == "GBP":
            return f"£{formatted_number}"
        elif currency == "JPY":
            return f"¥{formatted_number}"
        else:  # Default USD
            return f"${formatted_number}"
    
    def format_date(self, timestamp: float, locale: Optional[str] = None) -> str:
        """Format date according to locale conventions."""
        import datetime
        
        target_locale = locale or self.current_locale
        
        if target_locale not in self.locales:
            target_locale = self.default_locale
        
        locale_config = self.locales[target_locale]
        dt = datetime.datetime.fromtimestamp(timestamp)
        
        try:
            return dt.strftime(locale_config.date_format)
        except ValueError:
            # Fallback to ISO format
            return dt.strftime("%Y-%m-%d")
    
    def format_time(self, timestamp: float, locale: Optional[str] = None) -> str:
        """Format time according to locale conventions."""
        import datetime
        
        target_locale = locale or self.current_locale
        
        if target_locale not in self.locales:
            target_locale = self.default_locale
        
        locale_config = self.locales[target_locale]
        dt = datetime.datetime.fromtimestamp(timestamp)
        
        try:
            return dt.strftime(locale_config.time_format)
        except ValueError:
            # Fallback to 24-hour format
            return dt.strftime("%H:%M:%S")
    
    def get_supported_locales(self) -> List[Dict[str, str]]:
        """Get list of supported locales."""
        return [
            {
                "code": code,
                "name": config.name,
                "language": config.language,
                "country": config.country,
                "rtl": config.rtl
            }
            for code, config in self.locales.items()
        ]
    
    def get_translation_coverage(self) -> Dict[str, Any]:
        """Get translation coverage statistics."""
        coverage_stats = {
            "total_locales": len(self.locales),
            "total_keys": 0,
            "coverage_by_locale": {},
            "missing_translations": {}
        }
        
        # Find all translation keys across all locales
        all_keys = set()
        for translations in self.translations.values():
            all_keys.update(translations.keys())
        
        coverage_stats["total_keys"] = len(all_keys)
        
        # Calculate coverage per locale
        for locale_code in self.locales:
            if locale_code in self.translations:
                locale_keys = set(self.translations[locale_code].keys())
                coverage = len(locale_keys) / len(all_keys) if all_keys else 0
                missing_keys = all_keys - locale_keys
                
                coverage_stats["coverage_by_locale"][locale_code] = {
                    "coverage_percent": coverage * 100,
                    "translated_keys": len(locale_keys),
                    "missing_keys": len(missing_keys)
                }
                
                if missing_keys:
                    coverage_stats["missing_translations"][locale_code] = list(missing_keys)
            else:
                coverage_stats["coverage_by_locale"][locale_code] = {
                    "coverage_percent": 0.0,
                    "translated_keys": 0,
                    "missing_keys": len(all_keys)
                }
        
        return coverage_stats
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get i18n performance metrics."""
        return {
            "current_locale": self.current_locale,
            "supported_locales": len(self.locales),
            "total_translations": sum(len(t) for t in self.translations.values()),
            "cache_size": len(self.translation_cache),
            "requests": self.translation_metrics["requests"],
            "cache_hit_rate": (
                self.translation_metrics["cache_hits"] / max(1, self.translation_metrics["requests"])
            ) * 100,
            "fallbacks_used": self.translation_metrics["fallbacks_used"],
            "missing_translations_count": len(self.translation_metrics["missing_translations"]),
            "missing_translation_keys": list(self.translation_metrics["missing_translations"])
        }
    
    def export_translations(self, locale: str) -> Dict[str, str]:
        """Export all translations for a specific locale."""
        return self.translations.get(locale, {}).copy()
    
    def import_translations(self, locale: str, translations: Dict[str, str]):
        """Import translations from external source."""
        self.add_translations(locale, translations)
        print(f"🌐 Imported {len(translations)} translations for {locale}")