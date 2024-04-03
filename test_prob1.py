from Wang_Chongrui_utils import CustomCSVData
import backtrader as bt


def test():
    print("Testing the customized data class ...")
    cerebro = bt.Cerebro()

    # Example of adding Type 1 data
    data1 = CustomCSVData(
        dataname='aapl.csv',
        csvformat='type1'
    )
    cerebro.adddata(data1)

    # Example of adding Type 2 data
    data2 = CustomCSVData(
        dataname='002054.XSHE.csv',
        csvformat='type2'
        # Set other necessary parameters
    )
    cerebro.adddata(data2)

    # Example of adding Type 3 data
    data3 = CustomCSVData(
        dataname='ERCOTDA_price.csv',
        csvformat='type3'
        # Set other necessary parameters
    )
    cerebro.adddata(data3)

    cerebro.run()

    print("All Tests passed !!!")


if __name__ == "__main__":
    test()