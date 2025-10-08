def test_function():
    try:
        if True:
            print("Inside if")
        print("After if")
    except Exception as e:
        print("Exception:", e)

test_function()
