import json
import unittest
from pathlib import Path
from unittest.mock import patch
from frontend.app import app, build_geospatial_payload, validator

class FrontendMapTests(unittest.TestCase):

    def test_missing_or_invalid_metadata_has_no_map_layers(self):
        for validation in (None, {}, {'metadata': {}}, {'metadata': 'invalid'}):
            with self.subTest(validation=validation):
                self.assertFalse(build_geospatial_payload(validation, {}, 'sar_analysis')['available'])

    def test_geographic_pair_preserves_coordinates_and_names(self):
        metadata = {'georeferenced': True, 'crs': 'EPSG:4326', 'bounds': [77, 30, 78, 31], 'width': 120, 'height': 120, 'bands': 2}
        result = build_geospatial_payload({'metadata': {'optical': metadata, 'sar': metadata}}, {}, 'sar_analysis')
        self.assertTrue(result['available'])
        self.assertEqual([layer['name'] for layer in result['layers']], ['optical', 'sar'])
        self.assertEqual(result['layers'][0]['bounds_wgs84'], [77, 30, 78, 31])
        self.assertEqual(result['visual_layers'], [])

    def test_projected_bounds_are_converted_to_degrees(self):
        validation = {'metadata': {'georeferenced': True, 'crs': 'EPSG:3857', 'bounds': [0, 0, 111319.49079327357, 111325.1428663851]}}
        result = build_geospatial_payload(validation, {}, 'sar_analysis')
        self.assertTrue(result['available'])
        for (actual, expected) in zip(result['layers'][0]['bounds_wgs84'], [0, 0, 1, 1]):
            self.assertAlmostEqual(actual, expected, places=5)
        with patch('frontend.app.transform_bounds', None):
            self.assertFalse(build_geospatial_payload(validation, {}, 'sar_analysis')['available'])

    def test_invalid_bounds_are_not_sent_to_leaflet(self):
        for bounds in ([0, 0, float('nan'), 1], [0, 0, 1], [10, 0, 5, 1], [0, 0, 200, 100]):
            with self.subTest(bounds=bounds):
                result = build_geospatial_payload({'metadata': {'georeferenced': True, 'crs': 'EPSG:4326', 'bounds': bounds}}, {}, 'sar_analysis')
                self.assertFalse(result['available'])

    def test_real_geotiff_metadata_and_api_response(self):
        path = Path(__file__).resolve().parents[1] / 'data/test/geotiff/sar_test.tif'
        validation = validator.validate_single_image(str(path), expected_modality='sar')
        self.assertTrue(validation['valid'])
        expected = build_geospatial_payload(validation, {}, 'sar_analysis')
        self.assertTrue(expected['available'])
        with path.open('rb') as uploaded:
            response = app.test_client().post('/api/analyze', data={'query': 'Analyze this SAR image', 'sar_image': (uploaded, 'sar_test.tif')})
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertTrue(result['success'])
        self.assertEqual(result['geospatial']['layers'], json.loads(json.dumps(expected['layers'])))

    def test_map_page_and_assets_are_served(self):
        client = app.test_client()
        for path in ('/', '/map-view', '/static/css/map.css', '/static/js/map.js'):
            with self.subTest(path=path):
                with client.get(path) as response:
                    self.assertEqual(response.status_code, 200)
        self.assertIn(b'href="/map-view"', client.get('/').data)
if __name__ == '__main__':
    unittest.main()
