class class_name1
  private var1
  public function function_name1
    return var2->input
  public function function_name2
    var2->input GET
var3 new class_name1
var4 var3->function_name1
var5 sprintf SQL_QUERY var4
var6 mysql_connect
mysql_select_db
echo var5
var7 mysql_query var5
loop var8 mysql_fetch_array var7
  print_r var8
  echo
mysql_close var6
