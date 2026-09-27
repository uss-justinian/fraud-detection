PRODUCTS = {
    "apple": 0.5,
    "bread": 2.25,
    "milk": 1.2,
    "coffee": 7.99,
    "cheese": 4.5,
}


def get_price(name):
    return PRODUCTS[name]


def list_products():
    return sorted(PRODUCTS.items())
