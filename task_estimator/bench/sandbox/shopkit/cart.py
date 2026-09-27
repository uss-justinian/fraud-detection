from shopkit.catalog import get_price


class Cart:
    def __init__(self):
        self.items = {}
        self.discount = 0.0

    def add_item(self, name, qty=1):
        self.items[name] = self.items.get(name, 0) + qty

    def remove_item(self, name):
        del self.items[name]

    def apply_discount(self, pct):
        self.discount = pct

    def subtotal(self):
        return sum(get_price(n) * q for n, q in self.items.items())

    def calc_total(self):
        total = self.subtotal() * (1 - self.discount)
        if self.discount:
            total = total * (1 - self.discount)
        return round(total, 2)
