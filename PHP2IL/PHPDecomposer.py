from .php_utils import   remove_comments, \
                        get_php_snippets, \
                        php_line_processing

class PHPDecomposer:
    def __init__(self, php_code, file_id=None):
        self.php_code = php_code
        self.file_id = file_id
        #self.php_status = ""
        #self.php_status = self.find_php_status()
        self.php_status = None
        self.php_tokens = []
        self.php_tokens = self.tokenize_php_code()
        return
    
    def find_php_status(self) -> str:
        # Safe or Unsafe
        status = self.php_code.split("\n")[2].split(" ")[0]
        return status

    def tokenize_php_code(self):

        snippet = get_php_snippets(self.php_code)

        lines = remove_comments(snippet[0]).splitlines()

        lines = list(filter(None, lines))

        codelines = php_line_processing(lines, self.php_status, self.file_id)

        self.php_tokens = codelines
        return self.php_tokens
    
    def get_php_tokens(self):
        return self.php_tokens

