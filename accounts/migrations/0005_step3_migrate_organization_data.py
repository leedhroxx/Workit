from django.db import migrations


def migrate_organization_text_to_fk(apps, schema_editor):
    User = apps.get_model('accounts', 'User')
    Organization = apps.get_model('accounts', 'Organization')

    seen = {}
    counter = 1
    for user in User.objects.all():
        text = (user.organization_legacy or '').strip()
        if not text:
            continue
        if text not in seen:
            org, _ = Organization.objects.get_or_create(
                name=text,
                defaults={'code': f'ORG{counter:03d}'},
            )
            seen[text] = org
            counter += 1
        user.organization = seen[text]
        user.save(update_fields=['organization'])


def reverse_noop(apps, schema_editor):
    pass  # 되돌릴 필요 없음


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0004_step2_add_organization_fk'),
    ]

    operations = [
        migrations.RunPython(migrate_organization_text_to_fk, reverse_noop),
    ]