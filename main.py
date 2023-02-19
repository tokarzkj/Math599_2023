import commands

if __name__ == '__main__':
    print("Please select one of the commands: dft graph, verify properties")
    command = input()

    if command == "dft graph":
        commands.dft_graph()
    elif command == "verify properties":
        commands.verify_dft_properties()
