var1 SESSION
class class_name1
  private var2
  public function function_name1 var3
    var4->data var3
  public function function_name2
    return var4->data
  public function function_name3
    var4->data mysql_real_escape_string var4->data
var5 new class_name1 var1
var5->function_name3
var1 var5->function_name2
var6 sprintf SQL_QUERY var1
var7 mysql_connect
mysql_select_db
echo var6
var8 mysql_query var6
loop var2 mysql_fetch_array var8
  print_r var2
  echo
mysql_close var7
