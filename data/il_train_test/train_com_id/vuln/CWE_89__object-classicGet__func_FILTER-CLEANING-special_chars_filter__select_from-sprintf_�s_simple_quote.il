class class_name1
  private var1
  public function function_name1
    return var2->input
  public function function_name2
    var2->input GET
var3 new class_name1
var4 var3->function_name1
var5 filter_var var4 FILTER_SANITIZE_SPECIAL_CHARS
var4 var5
var6 sprintf SQL_QUERY var4
var7 mysql_connect
mysql_select_db
echo var6
var8 mysql_query var6
loop var9 mysql_fetch_array var8
  print_r var9
  echo
mysql_close var7
