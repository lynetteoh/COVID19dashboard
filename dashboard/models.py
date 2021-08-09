from django.db import models

# Create your models here.
class Country(models.Model):
    Country_ID = models.CharField(max_length=10, primary_key=True)
    Country_Name = models.CharField(max_length=10)

    def __str__(self):
        return self.Country_Name

    class Meta:
        db_table = 'Country'
    


class State(models.Model):
    State_ID = models.CharField(max_length=10, primary_key=True)
    Country_ID = models.ForeignKey(Country, on_delete=models.CASCADE)
    State_Name = models.CharField(max_length=50)

    def __str__(self):
        return "%s %s %s" % (self.State_Name, self.State_ID, self.Country_ID.Country_ID)
    
    class Meta:
        db_table = 'State'


class District(models.Model):
    District_ID = models.CharField(max_length=10, primary_key=True)
    State_ID = models.ForeignKey(State, on_delete=models.CASCADE)
    District_Name = models.CharField(max_length=50)

    def __str__(self):
        return "%s %s %s" % (self.District_ID, self.District_Name, self.State_ID.State_ID)

    class Meta:
        db_table = 'District'
        


class Case(models.Model):
    # 1 - country level, 2 - state level, 3 - district level
    Ref_Key = models.PositiveIntegerField()
    Reference_ID = models.CharField(max_length=10)
    Date = models.DateField('Date Occured')
    Daily_infected = models.PositiveIntegerField(null=True)
    Daily_deaths = models.PositiveIntegerField(null=True)
    Daily_recoveries = models.PositiveIntegerField(null=True)
    Respiratory_aid = models.PositiveIntegerField(null=True)
    No_of_patient_in_ICU = models.PositiveIntegerField(null=True)
    Total_infected = models.PositiveIntegerField(null=True)
    Total_deaths = models.PositiveIntegerField(null=True)
    Total_recoveries = models.PositiveIntegerField(null=True)
    Active_cases = models.PositiveIntegerField(null=True)
    Total_tests = models.PositiveIntegerField(null=True)
    Is_actual = models.BooleanField(default=True)

    def __str__(self):
        return "%s %s %s %s %s %s %s" % (self.Ref_Key, self.Reference_ID, self.Date, self.Daily_infected, self.Total_infected, self.Active_cases, self.Is_actual)


    class Meta:
        db_table = 'Case'
        unique_together = (('Date', 'Reference_ID', 'Is_actual'),)
        ordering = ['-Date']
    

class Zone(models.Model):
    Ref_Key = models.PositiveIntegerField()
    Reference_ID = models.CharField(max_length=10)
    Date = models.DateField('Date Occured')
    Is_actual = models.BooleanField(default=True)
    Zone_colour = models.PositiveIntegerField()

    def __str__(self):
        return "%s %s %s %s %s" % (self.Ref_Key, self.Reference_ID, self.Date, self.Zone_colour, self.Is_actual)

    class Meta:
        db_table = 'Zone'
        unique_together = (('Date', 'Reference_ID', 'Is_actual'),)
        ordering = ['-Date']
    