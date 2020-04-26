import upscaler
import mask_remesher
import hdsutils
import createmask
import matplotlib.pyplot as plt

upscaler.rescaleMesh("/tmp/tmesh.ply", 64, 64, "/tmp/xmesh.ply")
#timg = hdsutils.loadHDSImage("/tmp/timage.hds")
#timg = upscaler.downsampleImage(timg, 4) 
#print timg.shape
#plt.imshow(timg.squeeze())
#plt.show()
#mask = createmask.renderMask("/home/sassy/thesis/thesis-689/temporary/tests/non-lambertian-tests/test-statue-multires/meshes/original.ply", 128, 128)
#mask_remesher.remesh("/home/sassy/thesis/thesis-689/temporary/tests/non-lambertian-tests/test-statue-multires/meshes/original.ply", omeshfile="./tempoutput.ply", W=128, H=128, mask=mask)
