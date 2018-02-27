import unittest

class TestAPI(unittest.TestCase):

    def test_call(self):
		input = "123abc"
		output = Score.run(input)
		self.assert(output, 'FOO')
            
if __name__ == '__main__':
    unittest.main()	