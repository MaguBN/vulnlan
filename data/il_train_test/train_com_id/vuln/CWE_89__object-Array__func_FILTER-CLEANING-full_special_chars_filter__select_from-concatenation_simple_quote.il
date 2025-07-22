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
var5 filter_var var4 FILTER_SANITIZE_FULL_SPECIAL_CHARS
var4 var5
var6 SQL_QUERY var4
var7 mysql_connect
mysql_select_db
echo var6
var8 mysql_query var6
loop var9 mysql_fetch_array var8
  print_r var9
  echo
mysql_close var7
