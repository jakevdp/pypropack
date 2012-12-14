all : inplace

inplace:
	python setup.py build_ext --inplace

test:
	nosetests pypropack