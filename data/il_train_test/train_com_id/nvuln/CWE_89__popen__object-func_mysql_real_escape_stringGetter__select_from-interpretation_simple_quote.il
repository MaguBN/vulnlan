var1 popen
var2 fread var1 4096
pclose var1
class class_name1
  private var3
  public function function_name1 var4
    var5->data var4
  public function function_name2
    return var5->data
  public function function_name3
    var5->data mysql_real_escape_string var5->data
var6 new class_name1 var2
var6->function_name3
var2 var6->function_name2
var7 SQL_QUERY var2
var8 mysql_connect
mysql_select_db
echo var7
var9 mysql_query var7
loop var3 mysql_fetch_array var9
  print_r var3
  echo
mysql_close var8
