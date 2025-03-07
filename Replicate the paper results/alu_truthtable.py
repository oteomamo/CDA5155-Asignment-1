def alu_operation(a, b, opcode):
    """
    a, b are 2-bit integers (0 to 3)
    opcode is 2-bit integer (0 to 3),
    returns a 2-bit integer (0 to 3) ignoring carry for addition.
    """
    if opcode == 0:  # 00 => AND
        return a & b
    elif opcode == 1:  # 01 => OR
        return a | b
    elif opcode == 2:  # 10 => ADD (mod 4)
        return (a + b) % 4
    else:  # 3 => XOR
        return a ^ b

def generate_alu_truth_table():
    print(" A |  B | Op | Output ")
    print("----------------------")
    for A in range(4):    # 0..3 for 2-bit A
        for B in range(4):  # 0..3 for 2-bit B
            for op in range(4):  # 0..3 for opcode
                out = alu_operation(A, B, op)
                # Convert to 2-bit binary strings
                A_bin  = format(A, '02b')
                B_bin  = format(B, '02b')
                op_bin = format(op, '02b')
                out_bin= format(out, '02b')
                print(f"{A_bin} | {B_bin} | {op_bin} |  {out_bin}")

if __name__ == "__main__":
    generate_alu_truth_table()
