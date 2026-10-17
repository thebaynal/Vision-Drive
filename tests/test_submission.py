from scripts.check_submission import check_links, check
from scripts.export_evaluation import retain_supplementary_report


def test_document_links_check_headings_as_well_as_file_existence(tmp_path):
    target = tmp_path/'guide.md'
    target.write_text('# Setup guide\n\n<a id="manual-anchor"></a>\n', encoding='utf-8')
    assert not check_links(tmp_path/'README.md', '[a](guide.md#setup-guide) [b](guide.md#manual-anchor)')
    assert len(check_links(tmp_path/'README.md', '[a](guide.md#missing) [b](deleted.md)')) == 2


def test_person_report_export_keeps_object_benchmark_without_duplication():
    marker = '## Bottle and cup detection: additional measured benchmark'
    previous = '# Old person report\n\n' + marker + '\nGenuine bottle/cup results.\n'
    exported = retain_supplementary_report('# New person report\n', previous)
    assert exported.startswith('# New person report')
    assert 'Genuine bottle/cup results.' in exported
    assert retain_supplementary_report(exported, previous) == exported


def test_submission_links_notebooks_and_measured_tables_are_consistent():
    errors, _, _ = check()
    assert not errors, errors
