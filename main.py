import commands
import convolution

if __name__ == '__main__':
    print("Please select one of the commands: dft graph, verify properties, convolution")
    command = input()

    if command == "dft graph":
        commands.dft_graph()
    elif command == "verify properties":
        commands.verify_dft_properties()
    elif command == "convolution":
        convolution.convolution_command()
