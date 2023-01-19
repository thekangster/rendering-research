import drjit as dr
import mitsuba as mi

import matplotlib.pyplot as plt

mi.set_variant('llvm_ad_rgb')

scene = mi.load_file('scenes/cbox.xml', res=128, integrator='prb')
image_ref = mi.render(scene, spp=512)

# use traverse to pick the parameter to optimize
params = mi.traverse(scene)
key = 'red.reflectance.value'

# save the original red color 
param_ref = mi.Color3f(params[key])

# set to another blue then update the scene
params[key] = mi.Color3f(0.01, 0.2, 0.9)
params.update()

image_init = mi.render(scene, spp=128)

# preview image
# plt.axis('off')
# plt.imshow(mi.util.convert_to_bitmap(image_init))
# plt.show()

# optimize scene parameter with a learning rate of 0.05
# set the color to optimize on the optimizer
# update values back to params data structure
opt = mi.ad.Adam(lr=0.05)
opt[key] = params[key]
params.update(opt)

# mean square error, or L_2 error
# average of the squares of errors between current image and reference
def mse(image):
    return dr.mean(dr.sqr(image - image_ref))

iteration_count = 50
errors = []
for it in range(iteration_count):
    # Perform a (noisy) differentiable rendering of the scene
    image = mi.render(scene, params, spp=4)

    # Evaluate the objective function from the current rendered image
    loss = mse(image)

    # Backpropogate through the rendering process
    dr.backward(loss)

    # Optimizer: take a gradient descent step
    opt.step()

    # Post-process the optimized parameters to ensure legal color values.
    opt[key] = dr.clamp(opt[key], 0.0, 1.0)

    # Update the scene state to the new optimized values
    params.update(opt)

    # Track the difference betrween the current color and the true value
    err_ref = dr.sum(dr.sqr(param_ref - params[key]))
    print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}", end='\r')
    errors.append(err_ref)
print('\nOptimization complete.')

image_final = mi.render(scene, spp=128)

plt.axis('off')
plt.imshow(mi.util.convert_to_bitmap(image_final))
plt.show()

