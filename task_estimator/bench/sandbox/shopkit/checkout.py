from shopkit.cart import Cart


def checkout(order_lines, discount=0.0):
    """order_lines: list of (name, qty)"""
    cart = Cart()
    for name, qty in order_lines:
        cart.add_item(name, qty)
    cart.apply_discount(discount)
    return {"lines": dict(cart.items), "total": cart.calc_total()}


def receipt(order_lines, discount=0.0):
    result = checkout(order_lines, discount)
    lines = [f"{n} x{q}" for n, q in result["lines"].items()]
    lines.append(f"TOTAL {result['total']}")
    return "\n".join(lines)
