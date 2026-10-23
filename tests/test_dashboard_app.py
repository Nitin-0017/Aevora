"""Streamlit app interactions; browser screenshots are verified separately."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def widget(elements,label):
    return next(item for item in elements if item.label==label)


class DashboardAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            raise unittest.SkipTest('Streamlit required')
        if not (ROOT/'artifacts/reports/FD001/performance_audit/audit.json').exists():
            raise unittest.SkipTest('Local audited artifacts required')
        cls.AppTest=AppTest

    @classmethod
    def tearDownClass(cls):
        from streamlit.testing.v1.app_test import TMP_DIR
        TMP_DIR.cleanup()

    def test_operator_workflow_models_conversion_and_stale_results(self):
        app=self.AppTest.from_file(str(ROOT/'dashboard.py'),default_timeout=90).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value,'Unavailable')
        for model,expected in [('baseline','95.00'),('ridge','106.03'),('gradient_boosting','122.09'),('lstm','131.05')]:
            app.sidebar.selectbox[0].set_value(model).run()
            widget(app.button,'Run prediction').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value,expected+' cycles')
        self.assertTrue(any('Prediction generated:' in item.value for item in app.caption))
        self.assertTrue(any('not a live industrial machine' in item.value for item in app.caption))
        self.assertTrue(any('Why are we showing this?' in item.value for item in app.markdown))
        widget(app.checkbox,'I have a verified average operating rate').check().run()
        widget(app.number_input,'Verified operating rate (cycles per hour)').set_value(20.).run()
        self.assertTrue(any('operating hours' in item.value and '6.55' in item.value for item in app.markdown))
        widget(app.selectbox,'Select a demonstration engine').set_value(5).run()
        self.assertEqual(app.metric[0].value,'Unavailable')
        widget(app.button,'Run prediction').click().run()
        self.assertNotEqual(app.metric[0].value,'Unavailable')
        app.sidebar.slider[0].set_value(20).run()
        self.assertEqual(app.metric[0].value,'Unavailable')
        app.sidebar.number_input[0].set_value(120.).run()
        self.assertTrue(any('threshold must be below' in item.value for item in app.error))
        self.assertFalse(app.exception)

    def test_technical_pages_preserved(self):
        app=self.AppTest.from_file(str(ROOT/'dashboard.py'),default_timeout=90).run()
        app.sidebar.radio[0].set_value('Technical details').run()
        self.assertEqual(app.metric[0].value,'20')
        for page in ['Engine explorer','RUL prediction','Model comparison','Reliability insights','Model & data information']:
            widget(app.selectbox,'Technical section').set_value(page).run()
            self.assertFalse(app.exception,page)
        widget(app.selectbox,'Technical section').set_value('Engine explorer').run()
        widget(app.selectbox,'Engine').set_value(5).run()
        app.multiselect[0].set_value([]).run()
        self.assertTrue(any('Select at least one sensor' in item.value for item in app.info))

    def test_missing_artifact_error_state(self):
        from unittest.mock import patch
        from aevora.dashboard.data import ArtifactError
        with patch('aevora.dashboard.app.snapshot_resource',side_effect=ArtifactError('Missing local artifact')):
            app=self.AppTest.from_file(str(ROOT/'dashboard.py'),default_timeout=90).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Unable to assess' in item.value for item in app.error))
        self.assertTrue(any('Missing local artifact' in item.value for item in app.code))
