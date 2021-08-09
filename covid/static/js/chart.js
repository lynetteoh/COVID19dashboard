function infectedCasesChart(all_cases){
    console.log("generating chart");
    console.log(JSON.parse(all_cases)[0].fields.Date);
    var ctx = document.getElementById('infectedChart');
    var date = [];
    var total_infected = []
    var c;
    all_cases = JSON.parse(all_cases);
    for (c of all_cases)
    {
        date.push(c.fields.Date);
        total_infected.push(c.fields.Total_infected)
    }

    drawChart(date, total_infected, ctx, "Total Infected")
    
}

function deathCasesChart(all_cases){
    console.log("generating chart");
    console.log(JSON.parse(all_cases)[0].fields.Total_deaths);
    var ctx = document.getElementById('deathChart');
    var date = [];
    var total_death = []
    var c;
    all_cases = JSON.parse(all_cases);
    for (c of all_cases)
    {
        date.push(c.fields.Date);
        total_death.push(c.fields.Total_deaths)
    }

    drawChart(date, total_death, ctx, "Total Deaths")
}


function recoverCasesChart(all_cases){
    console.log("generating chart");
    console.log(JSON.parse(all_cases)[0].fields.Total_recoveries);
    var ctx = document.getElementById('recoveredChart');
    var date = [];
    var total_recoveries = []
    var c;
    all_cases = JSON.parse(all_cases);
    for (c of all_cases)
    {
        date.push(c.fields.Date);
        total_recoveries.push(c.fields.Total_recoveries)
    }
   
    drawChart(date, total_recoveries, ctx, "Total Recoveries")
}

function drawChart(date, cases, ctx, Label){
    var myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: date,
            datasets: [{
                label: Label,
                data: cases,
                fill:true,
                borderColor: ['rgba(75, 192, 192, 1)'],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.2)'
                ]                        
                
            }]
        },
        options: {
            scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }]
            }
        }
    });

}



function dailyInfectedCasesChart(all_cases, forecast){
    console.log("generating chart");
    var ctx = document.getElementById('daily_infected');
    var date = [];
    var daily_infected = []
    var c;
    all_cases = JSON.parse(all_cases);
    for (c of all_cases)
    {
        date.push(c.fields.Date);
        daily_infected.push(c.fields.Daily_infected)
    }

    forecast = JSON.parse(forecast);
    for (c of forecast)
    {
        date.push(c.fields.Date);
        daily_infected.push(c.fields.Daily_infected)
    }

    var data_actual = {
        label: "Daily infected",
        data: daily_infected,
        fill: true,
        borderColor: [
            'rgba(75, 192, 192, 1)'
        ], 
        backgroundColor: [
            'rgba(75, 192, 192, 0.2)'
        ], 
        pointBackgroundColor: ['rgba(75, 192, 192, 0.2)']   
    };
    

    
    var myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: date,
            datasets: [data_actual
            ]
        },
        options: {
            responsive: true,
            scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }]
            }
        }
    });

    var dataPoints = [];
    for(var i = daily_infected.length-1 ; i > daily_infected.length-8; i--){
        myChart.data.datasets[0].pointBackgroundColor[i] = 'red';
        dataPoints.push({x: i, y: daily_infected[i]});
    }

    myChart.data.datasets.data = dataPoints;
    myChart.update();
    
}


