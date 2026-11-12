import subprocess
import sys
import unittest
from pathlib import Path

class TestXor(unittest.TestCase):
    def test_entrenamiento(self):
        ruta = Path(__file__).resolve().parents[1] / 'neural_xor.py'
        p = subprocess.run([sys.executable, str(ruta)], capture_output=True, text=True, check=True)
        self.assertIn('¡red entrenada correctamente!', p.stdout)
        self.assertEqual(p.stdout.count('(esperado:'), 4)
