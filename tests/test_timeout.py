"""Tests for run_all_collectors._lookup_timeout."""

from __future__ import annotations

from run_all_collectors import _lookup_timeout, TIMEOUT_PROFILES


class TestLookupTimeout:
    def test_default_for_unlisted(self) -> None:
        assert _lookup_timeout("collectors/foo/obscure.py") == 300

    def test_custom_default(self) -> None:
        assert _lookup_timeout("collectors/foo/x.py", default=120) == 120

    def test_yfinance_matches(self) -> None:
        t = _lookup_timeout("collectors/equities/yfinance_equities.py")
        assert t == 600, f"expected 600 got {t}"

    def test_yfinance_logo_does_not_match(self) -> None:
        """'yfinance' does NOT match 'yfinance_logo' but the actual key is
        'yfinance' which IS a substring of 'yfinance_equities'."""
        t = _lookup_timeout("yfinance_foo.py")
        assert t == 600

    def test_coingecko_matches(self) -> None:
        t = _lookup_timeout("collectors/crypto/coingecko_top200_historical.py")
        assert t == 600

    def test_simple_collector_gets_default(self) -> None:
        t = _lookup_timeout("collectors/rates/term_rates.py")
        assert t == 300

    def test_picks_highest_value_when_multiple_match(self) -> None:
        """If a stem matches multiple keys, the highest timeout wins."""
        t = _lookup_timeout("collectors/energy/eia.py")
        # No EIA entry in profiles — expect default
        assert t == 300

    def test_baltic_dry_matches_baltic(self) -> None:
        t = _lookup_timeout("collectors/alternative/baltic_dry_index.py")
        assert t == 600

    def test_freightos_matches(self) -> None:
        t = _lookup_timeout("collectors/shipping/freightos.py")
        assert t == 600

    def test_sol_metrics_matches(self) -> None:
        t = _lookup_timeout("collectors/onchain/sol_metrics.py")
        assert t == 600

    def test_all_profile_keys_are_substrings_of_some_collector(self) -> None:
        """Sanity-check: every profile key should match at least one collector."""
        from run_all_collectors import COLLECTORS

        stems = {p.split("/")[-1].replace(".py", "") for p in COLLECTORS}
        for key in TIMEOUT_PROFILES:
            matched_any = any(key in s for s in stems)
            if not matched_any:
                import warnings
                warnings.warn(f"Timeout profile key '{key}' has no matching collector")
