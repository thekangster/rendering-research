import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant('scalar_rgb')

img = mi.render(mi.load_dict(mi.cornell_box()))

plt.axis('off')
plt.imshow(img ** (1.0 / 2.2))
plt.show()

