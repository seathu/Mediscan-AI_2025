



def get_new_valuenwdt(value):
    if value <= 650:
        height = value
    elif value > 650 and value <= 800:
        height = value * 0.8
    elif value >= 800 and value <= 1000:
        height = value * 0.5
    elif value >= 1000 and value <= 1200:
        height = value * 0.65
    elif value >= 1200 and value <= 1800:
        height = 1000    

    return int(height)
print(get_new_valuenwdt(351))
print(get_new_valuenwdt(651))
print(get_new_valuenwdt(1100))
print(get_new_valuenwdt(1700))