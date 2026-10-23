"""Streamlit's application harness executes real interactions, not a browser render."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class DashboardAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            raise unittest.SkipTest('Streamlit is required for application smoke checks')
        if not (ROOT/'artifacts/reports/FD001/performance_audit/audit.json').exists():
            raise unittest.SkipTest('Local audited artifacts required for real dashboard smoke check')
        cls.AppTest=AppTest

    @classmethod
    def tearDownClass(cls):
        # Streamlit 1.54 creates a module-level temporary directory on import.
        # Explicit cleanup avoids its shutdown ResourceWarning under -W error.
        from streamlit.testing.v1.app_test import TMP_DIR
        TMP_DIR.cleanup()

    def test_pages_models_and_interactions(self):
        app=self.AppTest.from_file(str(ROOT/'dashboard.py'),default_timeout=90).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value,'20')
        for page in ['Engine explorer','RUL prediction','Model comparison','Reliability insights','Model & data information']:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception,page)
        app.sidebar.radio[0].set_value('RUL prediction')
        for model in ['baseline','ridge','gradient_boosting','lstm']:
            app.sidebar.selectbox[0].set_value(model).run()
            self.assertFalse(app.exception,model)
            self.assertTrue(app.success)
        app.sidebar.slider[0].set_value(20).run()
        app.sidebar.button[0].click().run()
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value('Engine explorer').run()
        next(widget for widget in app.selectbox if widget.label == 'Engine').set_value(5).run()
        app.multiselect[0].set_value([]).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Select at least one sensor' in item.value for item in app.info))
        app.sidebar.number_input[0].set_value(120.).run()
        self.assertTrue(any('threshold must be below' in item.value for item in app.error))
        self.assertFalse(app.exception)

    def test_missing_artifact_error_state(self):
        from unittest.mock import patch
        from aevora.dashboard.data import ArtifactError
        with patch('aevora.dashboard.app.snapshot_resource',side_effect=ArtifactError('Missing local artifact: restore audited bundle')):
            app=self.AppTest.from_file(str(ROOT/'dashboard.py'),default_timeout=90).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Missing local artifact' in item.value for item in app.error))
