from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0005_step3_migrate_organization_data'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='user',
            name='organization_legacy',
        ),
    ]