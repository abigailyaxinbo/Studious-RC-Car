#include <linux/module.h>
#include <linux/init.h>
#include <linux/gpio.h>
#include <linux/interrupt.h>
#include <linux/jiffies.h>
/* Meta Information */
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Johannes 4 GNU/Linux");
MODULE_DESCRIPTION("A simple LKM for a gpio interrupt");

/** variable contains pin number o interrupt controller to which GPIO 17 is mapped to */
unsigned int irq_number;

long start_time, old_time;
static long elapsed;

module_param(elapsed, long, S_IRUGO);


/**
 * @brief Interrupt service routine is called, when interrupt is triggered
 */
static irq_handler_t gpio_irq_handler(unsigned int irq, void *dev_id, struct pt_regs *regs) {
         start_time = jiffies;
         elapsed = (start_time - old_time) * 1000 / CLOCKS_PER_SEC; // elapsed in milliseconds
         old_time = start_time;
	if (elapsed < 5){
		return IRQ_HANDLED;
	}
         //elapsed_time = ktime_to_ns(elapsed) / 100000;
         printk("Elapsed: %lu milliseconds\n", elapsed);
         return (irq_handler_t) IRQ_HANDLED;

}

/**
 * @brief This function is called, when the module is loaded into the kernel
 */
static int __init ModuleInit(void) {
	printk("qpio_irq: Loading module... ");
	old_time = jiffies;

	/* Setup the gpio */
	if(gpio_request(17, "rpi-gpio-17")) {
		printk("Error!\nCan not allocate GPIO 17\n");
		return -1;
	}

	/* Set GPIO 17 direction */
	if(gpio_direction_input(17)) {
		printk("Error!\nCan not set GPIO 17 to input!\n");
		gpio_free(17);
		return -1;
	}

	/* Setup the interrupt */
	irq_number = gpio_to_irq(17);

	if(request_irq(irq_number, (irq_handler_t) gpio_irq_handler, IRQF_TRIGGER_RISING, "my_gpio_irq", NULL) != 0){
		printk("Error!\nCan not request interrupt nr.: %d\n", irq_number);
		gpio_free(17);
		return -1;
	}

	printk("Done!\n");
	printk("GPIO 17 is mapped to IRQ Nr.: %d\n", irq_number);
	return 0;
}

/**
 * @brief This function is called, when the module is removed from the kernel
 */
static void __exit ModuleExit(void) {
	printk("gpio_irq: Unloading module... ");
	free_irq(irq_number, NULL);
	gpio_free(17);

}

module_init(ModuleInit);
module_exit(ModuleExit);
