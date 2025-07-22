class class_name1
  private var1
  public function function_name1
    return var2->input
  public function function_name2
    var2->input GET
var3 new class_name1
var4 var3->function_name1
class class_name2
  private var5
  public function function_name2 var1
    var2->data var1
  public function function_name3
    return var2->data
  public function function_name4
    var2->data mysql_real_escape_string var2->data
var6 new class_name2 var4
var6->function_name4
var4 var6->function_name3
var7 sprintf SQL_QUERY var4
var8 mysql_connect
mysql_select_db
echo var7
var9 mysql_query var7
loop var5 mysql_fetch_array var9
  print_r var5
  echo
mysql_close var8
