class class_name1
  private var1
  public function function_name1
    return var2->input_1
  public function function_name2
    var2->input array
    var2->input_0
    var2->input_1 GET
    var2->input_2
var3 new class_name1
var4 var3->function_name1
var4 mysql_real_escape_string var4
var5 sprintf SQL_QUERY var4
var6 mysql_connect
mysql_select_db
echo var5
var7 mysql_query var5
loop var8 mysql_fetch_array var7
  print_r var8
  echo
mysql_close var6
