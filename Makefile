PWD=$(shell pwd)
KERNEL_BUILD=/lib/modules/$(shell uname -r)/build

obj-m+=speed_driver_no_tree.o

all:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean
