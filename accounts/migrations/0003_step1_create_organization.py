from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0002_user_current_session_key_user_failed_login_attempts_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Organization',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, unique=True, verbose_name='부서명')),
                ('code', models.CharField(max_length=20, unique=True, verbose_name='부서코드')),
                ('is_active', models.BooleanField(default=True, verbose_name='사용 여부')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={'verbose_name': '부서', 'verbose_name_plural': '부서 목록', 'ordering': ['name']},
        ),
        migrations.AddField(
            model_name='user',
            name='notification_enabled',
            field=models.BooleanField(default=True, verbose_name='알림 수신'),
        ),
        migrations.RenameField(
            model_name='user',
            old_name='organization',
            new_name='organization_legacy',
        ),
    ]