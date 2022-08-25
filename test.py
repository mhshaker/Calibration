import UncertaintyM as uncM


x = [4,3,2,1]
y = [1,2,3,4]

comp, comp_p = uncM.order_comparison([x], [y])
print(comp)