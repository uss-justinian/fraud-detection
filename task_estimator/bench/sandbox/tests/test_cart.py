from shopkit.cart import Cart


def test_subtotal():
    c = Cart()
    c.add_item("apple", 4)
    c.add_item("bread")
    assert c.subtotal() == 4.25


def test_total_no_discount():
    c = Cart()
    c.add_item("milk", 2)
    assert c.calc_total() == 2.4


def test_total_with_discount():
    c = Cart()
    c.add_item("coffee", 1)
    c.apply_discount(0.5)
    assert c.calc_total() == 4.0
