class Color():
    
    BLACK           = '\033[30m'    # (letter)black
    RED             = '\033[31m'    # (letter)red
    GREEN           = '\033[32m'    # (letter)green
    YELLOW          = '\033[33m'    # (letter)yellow
    BLUE            = '\033[34m'    # (letter)blue
    MAGENTA         = '\033[35m'    # (letter)magenta
    CYAN            = '\033[36m'    # (letter)cyan
    WHITE           = '\033[37m'    # (letter)white
    COLOR_DEFAULT   = '\033[39m'    # reset the letter color to default
    BOLD            = '\033[1m'     # bold letter
    UNDERLINE       = '\033[4m'     # underline
    INVISIBLE       = '\033[08m'    # invisible letter
    REVERCE         = '\033[07m'    # reverse letter color and background color
    BG_BLACK        = '\033[40m'    # (background)black
    BG_RED          = '\033[41m'    # (background)red
    BG_GREEN        = '\033[42m'    # (background)green
    BG_YELLOW       = '\033[43m'    # (background)yellow
    BG_BLUE         = '\033[44m'    # (background)blue
    BG_MAGENTA      = '\033[45m'    # (background)magenta
    BG_CYAN         = '\033[46m'    # (background)cyan
    BG_WHITE        = '\033[47m'    # (background)white
    BG_DEFAULT      = '\033[49m'    # reset the background color to default
    RESET           = '\033[0m'     # reset all to default


class InputFunctions():

    @staticmethod
    def check_validity(input_str: str, acceptable_str_list: list):

        validity = True if input_str in acceptable_str_list else False

        return validity
