TF_LFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TF_CFLAGS:=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')

CXX_FLAGS:= -g

all: mitsuba_v2_op.so mitsuba_v2_grad_op.so mitsuba_v2_sparse_op.so mitsuba_v2_sparse_grad_op.so mitsuba_v2_vtx_grad_op.so mitsuba_v2_vtx_op.so mitsuba_v2_reductor_grad_op.so mitsuba_v2_vtx_stacked_op.so mitsuba_v2_vtx_reductor_stacked_grad.so gentex install
mitsuba_v2_op.so: mitsuba_op.cc
	g++ -std=c++11 -shared mitsuba_op.cc -o mitsuba_v2_op.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} $(CXX_FLAGS)
mitsuba_v2_grad_op.so: mitsuba_grad_op.cc
	g++ -std=c++11 -shared mitsuba_grad_op.cc -o mitsuba_v2_grad_op.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} $(CXX_FLAGS)

mitsuba_v2_sparse_op.so: mitsuba_sparse_op.cc tf_utils.h
	g++ -std=c++11 -shared mitsuba_sparse_op.cc -o mitsuba_v2_sparse_op.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} $(CXX_FLAGS)

mitsuba_v2_sparse_grad_op.so: mitsuba_sparse_grad_op.cc tf_utils.h
	g++ -std=c++11 -shared mitsuba_sparse_grad_op.cc -o mitsuba_v2_sparse_grad_op.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} $(CXX_FLAGS)

#mitsuba_v2_reductor_op.so: reductor/mitsuba_reductor_op.cc tf_utils.h
#	g++ -std=c++11 -shared mitsuba_reductor_op.cc -o mitsuba_v2_reductor_op.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} $(CXX_FLAGS)

mitsuba_v2_reductor_grad_op.so: vtx/reductor/mitsuba_vtx_reductor_grad_op.cc vtx/ply_utils.h
	g++ -std=c++11 -shared vtx/reductor/mitsuba_vtx_reductor_grad_op.cc -o mitsuba_v2_reductor_grad_op.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -Ivtx/ $(CXX_FLAGS)

mitsuba_v2_vtx_op.so: vtx/mitsuba_vtx_op.cc vtx/ply_utils.h
	g++ -std=c++11 -shared vtx/mitsuba_vtx_op.cc -o mitsuba_v2_vtx_op.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} $(CXX_FLAGS)
mitsuba_v2_vtx_grad_op.so: vtx/mitsuba_vtx_grad_op.cc vtx/ply_utils.h
	g++ -std=c++11 -shared vtx/mitsuba_vtx_grad_op.cc -o mitsuba_v2_vtx_grad_op.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} $(CXX_FLAGS)

mitsuba_v2_vtx_reductor_stacked_grad.so: vtx/reductor/mitsuba_vtx_reductor_stacked_grad.cc vtx/ply_utils.h
	g++ -std=c++11 -shared vtx/reductor/mitsuba_vtx_reductor_stacked_grad.cc -o mitsuba_v2_reductor_stacked_grad.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -Ivtx/ $(CXX_FLAGS)

mitsuba_v2_vtx_stacked_op.so: vtx/mitsuba_vtx_stacked_op.cc vtx/ply_utils.h
	g++ -std=c++11 -shared vtx/mitsuba_vtx_stacked_op.cc -o mitsuba_v2_vtx_stacked_op.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} $(CXX_FLAGS)
gentex: gen_textures.cpp
	g++ -std=c++11 gen_textures.cpp -o gentex

.PHONY: clean install
clean:
	rm ./*.so
install:
	cp ./*.so ../qdispatch/
	#cp ./build/*.so ../qdispatch/
