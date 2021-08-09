function check_date(form){
    console.log("check date")
    var startDate = document.getElementById("startDate").valueAsDate;
    var endDate = document.getElementById("endDate").valueAsDate;
    console.log(form)
    if (startDate > endDate){
        window.alert("End date is after start date. Please select another date!");
    }
    else {
        document.getElementById(form).submit();
    }

}
