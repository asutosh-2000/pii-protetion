from django.db import models
import random

# class Userlist(models.Model):
#     manager_id = models.BigIntegerField(primary_key=True)
#     manager_username = models.CharField(max_length=255)
#     manager_password = models.CharField(max_length=255)

#     def save(self, *args, **kwargs):
#         if not self.manager_id:
#             self.manager_id = random.randint(10**14, 10**15-1)  # 15-digit random number
#         super().save(*args, **kwargs)

#     class Meta:
#         db_table = 'manager'  # Specify the existing MySQL table

#     def __str__(self):
#         return self.manager_username

class Manager(models.Model):
    manager_id = models.BigIntegerField(primary_key=True)
    manager_username = models.CharField(max_length=255)
    manager_password = models.CharField(max_length=255)

    def save(self, *args, **kwargs):
        if not self.manager_id:
            self.manager_id = random.randint(10**14, 10**15-1) 
        super().save(*args, **kwargs)

    class Meta:
        db_table = 'manager'  

    def __str__(self):
        return self.manager_username


