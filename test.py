import numpy as np

from pprint import pprint

_4 = np.identity(4)
print(f'4X4{_4.shape}')
pprint(_4)

_4x1 = np.ones((4,1))
print(f'4X1{_4x1.shape}')
pprint(_4x1)

print()
pprint(_4 @ _4x1)