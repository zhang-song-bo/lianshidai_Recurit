class QuantumDimensionError(Exception):
    """自定义异常类：维度不匹配时抛出"""
    pass

class QuantumMatrix:
    def __init__(self, matrix):
        """
        初始化量子矩阵，格式应为：
        [[(数值, 字符), (数值, 字符)],
         [(数值, 字符), (数值, 字符)]]
        """
        self.matrix = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0])

    def multiply(self, other: "QuantumMatrix") -> "QuantumMatrix":
        """执行矩阵乘法"""
        if self.cols != other.rows:
            raise QuantumDimensionError("矩阵维度不匹配，无法相乘")

        # 计算结果矩阵的大小 (self.rows x other.cols)
        result = [[(0, ' ') for _ in range(other.cols)] for _ in range(self.rows)]

        for i in range(self.rows):  # 遍历 A 的行
            for j in range(other.cols):  # 遍历 B 的列
                # 保存二元组的数据信息
                sum_value = 0
                max_char = 0
                for k in range(self.cols):  # 遍历 A 的列 / B 的行
                    a_value, a_phase = self.matrix[i][k]
                    b_value, b_phase = other.matrix[k][j]

                    # 极化值和相位值计算
                    sum_value += a_value * b_value
                    max_char = max(max(ord(a_phase), ord(b_phase)), max_char)

                result[i][j] = (sum_value, chr(max_char))

        return QuantumMatrix(result)

    def __str__(self):
        # 格式化输出矩阵
        return "\n".join(str(row) for row in self.matrix)

if __name__ == '__main__':
    # 测试用例
    matrix_a = QuantumMatrix([
        [(1, 'a'), (2, 'b')],
        [(4, 'y'), (5, '6')]
    ])

    matrix_b = QuantumMatrix([
        [(-1, 'z'), (0, 'x')],
        [(0, 'm'), (1, 'n')],
        [(1, 'm'), (2, 'n')]
    ])

    result = matrix_a.multiply(matrix_b)
    print("相乘结果：")
    print(result)
